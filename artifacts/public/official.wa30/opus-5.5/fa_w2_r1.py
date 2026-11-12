import numpy as np

def build():
    P = {}; G = []
    def nd(i, op, ins, **kw):
        d = {"id": i, "op": op, "in": ins}; d.update(kw); G.append(d); return i
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    def put(K, k, dr, dc, v=1.0):
        K[k // 2 + dr, k // 2 + dc] += v
    nd("one", "slice", ["coords"], start=2, stop=3)
    for c in (1, 2, 3, 9, 11, 14):
        nd("c%d" % c, "slice", ["x0"], start=c, stop=c + 1)
    Wc0 = np.zeros((64, 3, 1, 1)); bc0 = np.zeros(64)
    for k in range(16):
        for q, (axis, off) in enumerate([(0, 1), (1, 1), (0, -1), (1, -1)]):
            Wc0[q * 16 + k, axis] = 126.0; bc0[q * 16 + k] = -8 * k + off
    P["Wc0"] = Wc0; P["bc0"] = bc0
    nd("sa", "conv", ["coords"], w="Wc0", b="bc0"); nd("sb", "relu", ["sa"])
    nd("S", "min2", ["sb", "one"])
    Wph = np.zeros((2, 64, 1, 1))
    for k in range(16):
        Wph[0, k] = 1; Wph[0, 32 + k] = -1; Wph[1, 16 + k] = 1; Wph[1, 48 + k] = -1
    P["Wph"] = Wph
    nd("ph", "conv", ["S"], w="Wph")
    nd("rp", "slice", ["ph"], start=0, stop=1); nd("cpz", "slice", ["ph"], start=1, stop=2)
    nd("A00", "mul", ["rp", "cpz"])
    Wbd = np.zeros((32, 64, 1, 1))
    for a in range(2):
        for i in range(16):
            Wbd[a * 16 + i, a * 16 + i] = 1
            if i < 15: Wbd[a * 16 + i, a * 16 + i + 1] = -1
    P["Wbd"] = Wbd
    nd("bands", "conv", ["S"], w="Wbd")
    Wcm = np.zeros((256, 32, 1, 1))
    for i in range(16):
        for j in range(16):
            Wcm[i * 16 + j, i] = 1; Wcm[i * 16 + j, 16 + j] = 1
    P["Wcm"] = Wcm; P["bcm"] = -np.ones(256)
    nd("cma", "conv", ["bands"], w="Wcm", b="bcm"); nd("CM", "relu", ["cma"])
    WN = np.zeros((4, 16, 3, 3)); bN = np.array([0., 0., -3., 0.])
    for (dr, dc) in DIRS:
        put(WN[0, 0], 3, dr, dc); put(WN[1, 2], 3, dr, dc)
        for c in (0, 3, 4, 5, 9): put(WN[2, c], 3, dr, dc)
        for c in (3, 4): put(WN[3, c], 3, dr, dc)
    P["WN"] = WN; P["bN"] = bN
    nd("Nc", "conv", ["x0"], w="WN", b="bN"); nd("NR", "relu", ["Nc"])
    nd("NC", "min2", ["NR", "one"])
    nd("n0c", "slice", ["NC"], start=0, stop=1); nd("n2c", "slice", ["NC"], start=1, stop=2)
    nd("fullc", "slice", ["NC"], start=2, stop=3); nd("n34c", "slice", ["NC"], start=3, stop=4)
    WX = np.zeros((2, 16, 1, 1))
    for c in (5, 11, 12, 15): WX[0, c] = 1
    WX[1, 15] = 1; WX[1, 11] = 1
    P["WX"] = WX
    nd("WA", "conv", ["x0"], w="WX")
    nd("Wl", "slice", ["WA"], start=0, stop=1); nd("AD", "slice", ["WA"], start=1, stop=2)
    nd("B9", "mul", ["c9", "n2c"])
    P["W7"] = np.ones((1, 1, 7, 7))
    nd("D1", "conv", ["B9"], w="W7"); nd("D1c", "min2", ["D1", "one"])
    nd("D2", "conv", ["D1c"], w="W7"); nd("dil6", "min2", ["D2", "one"])
    nd("TG2", "mul", ["c2", "dil6"]); nd("TGR", "max2", ["TG2", "B9"])
    nd("BCa", "mul", ["c9", "fullc"]); nd("CBc", "mul", ["BCa", "n0c"])
    nd("BC34", "mul", ["BCa", "n34c"])
    nd("X", "concat", ["Wl", "BCa", "CBc", "TGR", "c2", "c1", "c14", "c3", "AD", "BC34"])
    K = 10; P1 = np.zeros((K, K, 7, 7)); P2 = np.zeros((K, K, 7, 7))
    for c in range(K):
        for i in range(4):
            for j in range(4):
                P1[c, c, 3 + i, 3 + j] = 1; P2[c, c, 3 - i, 3 - j] = 1
    P["P1"] = P1; P["P2"] = P2
    nd("XS", "conv", ["X"], w="P1"); nd("XA", "mul", ["XS", "A00"])
    nd("XZ", "conv", ["XA"], w="P2"); nd("XC", "min2", ["XZ", "one"])
    names = ["Wc", "BX", "CB", "TGv", "C2", "C1", "PL", "F3", "ADc", "BX34"]
    for i, n in enumerate(names):
        nd(n, "slice", ["XC"], start=i, stop=i + 1)
    nd("BAR", "min2", ["C2", "C1"])
    P["Whalf"] = np.full((1, 1, 1, 1), 0.5)
    nd("hB", "conv", ["BAR"], w="Whalf"); nd("mval", "max2", ["TGv", "hB"])
    nd("pT", "mul", ["CM", "mval"]); nd("vT", "gmax", ["pT"])
    P["WnegM"] = -np.eye(256)
    nd("nmem", "dense", ["mem"], w="WnegM"); nd("dm", "vadd", ["vT", "nmem"])
    nd("dmr", "vrelu", ["dm"]); nd("mem_next", "vadd", ["mem", "dmr"])
    nd("qm", "mul", ["CM", "mem"])
    P["Wsum"] = np.ones((1, 256, 1, 1))
    nd("TMv", "conv", ["qm"], w="Wsum")
    P["W2x"] = np.full((1, 1, 1, 1), 2.0); P["bm1"] = np.array([-1.0])
    nd("Tma", "conv", ["TMv"], w="W2x", b="bm1"); nd("Tm", "relu", ["Tma"])
    nd("TM2", "conv", ["TMv"], w="W2x"); nd("TM2c", "min2", ["TM2", "one"])
    nd("Bm", "sub", ["TM2c", "Tm"]); nd("Bmr", "relu", ["Bm"])
    nd("T", "max2", ["TGv", "Tm"]); nd("BARa", "max2", ["BAR", "Bmr"])
    nd("UBa", "sub", ["BX34", "T"]); nd("UBb", "sub", ["UBa", "BARa"]); nd("UB", "relu", ["UBb"])
    nd("OBa", "sub", ["BX", "CB"]); nd("OB", "relu", ["OBa"])
    nd("GPa", "add", ["UB", "ADc"]); nd("GP", "min2", ["GPa", "one"])
    nd("OPa", "add", ["Wc", "BX"]); nd("OPb", "add", ["OPa", "BAR"]); nd("ObsP", "min2", ["OPb", "one"])
    nd("OPd", "sub", ["ObsP", "CB"]); nd("ObsPc", "relu", ["OPd"])
    nd("freeP", "sub", ["one", "ObsP"]); nd("freePc", "sub", ["one", "ObsPc"])
    nd("OBb", "add", ["Wc", "OB"]); nd("ObsB", "min2", ["OBb", "one"]); nd("freeB", "sub", ["one", "ObsB"])
    nd("GBa", "sub", ["T", "OB"]); nd("GB0", "relu", ["GBa"])
    nd("GBh", "conv", ["BARa"], w="Whalf"); nd("GB", "max2", ["GB0", "GBh"])
    nd("PA", "mul", ["PL", "A00"]); nd("CBA", "mul", ["CB", "A00"])
    nd("Q", "concat", ["freePc", "PA"])
    Wq1 = np.zeros((8, 2, 7, 7)); Wq2 = np.zeros((8, 8, 3, 3))
    for f in range(2):
        for d, (dr, dc) in enumerate(DIRS):
            put(Wq1[f * 4 + d, f], 7, 3 * dr, 3 * dc); put(Wq2[f * 4 + d, f * 4 + d], 3, dr, dc)
    P["Wq1"] = Wq1; P["Wq2"] = Wq2
    nd("Qa", "conv", ["Q"], w="Wq1"); nd("QS", "conv", ["Qa"], w="Wq2")
    nd("FPs", "slice", ["QS"], start=0, stop=4); nd("PAs", "slice", ["QS"], start=4, stop=8)
    nd("fB4", "mul", ["freeB", "FPs"]); nd("gB4", "mul", ["GB", "FPs"])
    nd("CBd", "mul", ["CBA", "PAs"])
    nd("free5", "concat", ["freeP", "fB4"]); nd("goal5", "concat", ["GP", "gB4"])
    gam = 0.99
    Wh = np.zeros((20, 5, 3, 3))
    for d, (dr, dc) in enumerate(DIRS):
        for f in range(5): put(Wh[d * 5 + f, f], 3, dr, dc, gam)
    P["Wh"] = Wh
    body = [{"id": "hs", "op": "conv", "in": ["h"], "w": "Wh"}]
    for d in range(4):
        body.append({"id": "hs%d" % d, "op": "slice", "in": ["hs"], "start": 5 * d, "stop": 5 * d + 5})
    body += [{"id": "hm1", "op": "max2", "in": ["hs0", "hs1"]},
             {"id": "hm2", "op": "max2", "in": ["hs2", "hs3"]},
             {"id": "hm", "op": "max2", "in": ["hm1", "hm2"]},
             {"id": "hf", "op": "mul", "in": ["hm", "free5"]},
             {"id": "hn", "op": "max2", "in": ["hf", "goal5"]}]
    G.append({"id": "H", "op": "loop", "T": 200,
              "state": [{"id": "h", "zeros": [5, 64, 64]}],
              "body": body, "next": {"h": "hn"}})
    WS3 = np.zeros((20, 5, 7, 7)); WS4 = np.zeros((20, 20, 3, 3))
    for d, (dr, dc) in enumerate(DIRS):
        for f in range(5):
            put(WS3[d * 5 + f, f], 7, 3 * dr, 3 * dc); put(WS4[d * 5 + f, d * 5 + f], 3, dr, dc)
    P["WS3"] = WS3; P["WS4"] = WS4
    nd("S3", "conv", ["H.h"], w="WS3"); nd("S4", "conv", ["S3"], w="WS4")
    nd("M20", "concat", ["PA", "CBd"] * 4)
    nd("prod", "mul", ["S4", "M20"]); nd("g", "gmax", ["prod"])
    nd("f3t", "mul", ["F3", "UB"]); nd("TB", "max2", ["T", "BARa"]); nd("cpt", "mul", ["CB", "TB"])
    nd("sc", "concat", ["CB", "f3t", "c11", "cpt"]); nd("s", "gmax", ["sc"])
    WvN = np.zeros((4, 20)); WvB = np.zeros((4, 20))
    for d in range(4):
        WvN[d, d * 5] = 1
        for f in range(1, 5): WvB[d, d * 5 + f] = 1
    P["WvN"] = WvN; P["WvB"] = WvB
    nd("vN", "dense", ["g"], w="WvN"); nd("vB", "dense", ["g"], w="WvB")
    nd("carry", "vslice", ["s"], start=0, stop=1)
    P["Wneg1"] = np.array([[-1.]]); P["b1"] = np.array([1.])
    nd("nc", "dense", ["carry"], w="Wneg1", b="b1")
    P["WnegI"] = -np.eye(4)
    nd("nvN", "dense", ["vN"], w="WnegI"); nd("diff", "vadd", ["vB", "nvN"])
    nd("cd", "vmul", ["diff", "carry"]); nd("mv", "vadd", ["vN", "cd"])
    P["Wa"] = np.array([[0., 1., 1., 0.]])
    nd("a5r", "dense", ["s"], w="Wa"); nd("a5n", "vmul", ["a5r", "nc"])
    nd("cp", "vslice", ["s"], start=3, stop=4); nd("cpc", "vmul", ["cp", "carry"])
    P["Wst"] = np.array([[0.02]])
    nd("cst", "dense", ["carry"], w="Wst")
    nd("a5x", "vadd", ["a5n", "cpc"]); nd("a5", "vadd", ["a5x", "cst"])
    nd("z", "vconcat", ["mv", "a5"])
    Wo = np.zeros((8, 5))
    for d in range(4): Wo[1 + d, d] = 10.0
    Wo[5, 4] = 20.0
    P["Wo"] = Wo; P["bo"] = np.array([-1., 0., 0., 0., 0., 0., -10., -10.])
    nd("logits", "dense", ["z"], w="Wo", b="bo")
    P["Wck"] = np.zeros((1, 1, 1, 1))
    nd("click", "conv", ["one"], w="Wck")
    params = {k: np.asarray(v, np.float32) for k, v in P.items()}
    return {"graph": G, "memory_size": 256, "params": params}

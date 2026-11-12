import numpy as np

def build():
    G = []; P = {}
    def N(i, op, ins, **kw):
        d = {"id": i, "op": op, "in": ins}; d.update(kw); G.append(d); return i
    def L(name, terms, bias=0.0, act=None):
        ids = [t[0] for t in terms]
        src = N(name + "_c", "vconcat", ids) if len(ids) > 1 else ids[0]
        P["w_" + name] = np.array([[t[1] for t in terms]], np.float32)
        P["b_" + name] = np.array([bias], np.float32)
        out = name if act is None else name + "_d"
        N(out, "dense", [src], w="w_" + name, b="b_" + name)
        if act == "sig":
            N(name, "vsigmoid", [out])
        elif act == "relu":
            N(name, "vrelu", [out])
        return name

    W = np.zeros((19, 16, 7, 7), np.float32); c = 3
    D = [(c + 1, c), (c - 1, c), (c, c + 1), (c, c - 1)]
    for gi, cols in enumerate([[8, 9, 11, 15], [1], [11]]):
        for k, (i, j) in enumerate(D):
            for col in cols:
                W[gi * 4 + k, col, i, j] = 1
    W[12, 12, c, c] = 1; W[12, 6, c, c] = 1
    W[13, 0, c, c] = 1; W[13, 14, c, c] = 1
    for t in range(7):
        if t != c:
            W[14, 11, c, t] = 1; W[15, 11, t, c] = 1
    W[16, 6, c, c] = 1; W[17, 9, c, c] = 1; W[18, 8, c, c] = 1
    P["WF"] = W
    N("F", "conv", ["x0"], w="WF")

    WE = np.zeros((4, 3, 1, 1), np.float32)
    WE[0, :, 0, 0] = [-63, 0, 1]; WE[1, :, 0, 0] = [63, 0, -62]
    WE[2, :, 0, 0] = [0, -63, 1]; WE[3, :, 0, 0] = [0, 63, -62]
    P["WE"] = WE
    N("Ec", "conv", ["coords"], w="WE"); N("E", "relu", ["Ec"])
    N("bar", "slice", ["F"], start=13, stop=14)
    N("EB", "mul", ["E", "bar"]); N("gEB", "gmean", ["EB"])
    P["Wg"] = np.eye(4, dtype=np.float32) * 128
    P["bg1"] = np.full(4, -1, np.float32); P["bg2"] = np.full(4, -2, np.float32)
    N("g1d", "dense", ["gEB"], w="Wg", b="bg1"); N("g1r", "vrelu", ["g1d"])
    N("g2d", "dense", ["gEB"], w="Wg", b="bg2"); N("g2r", "vrelu", ["g2d"])
    N("g12", "vconcat", ["g1r", "g2r"])
    I4 = np.eye(4, dtype=np.float32)
    P["Wgd"] = np.hstack([I4, -I4])
    N("g4", "dense", ["g12"], w="Wgd")
    P["Wvh"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], np.float32)
    N("gVH", "dense", ["g4"], w="Wvh")
    N("gV", "vslice", ["gVH"], start=0, stop=1); N("gH", "vslice", ["gVH"], start=1, stop=2)
    P["Wone"] = np.zeros((1, 8), np.float32); P["bone"] = np.ones(1, np.float32)
    N("one", "dense", ["avail"], w="Wone", b="bone")
    N("f12", "vconcat", ["g4", "g4", "g4"])
    N("Fs", "slice", ["F"], start=0, stop=12)
    N("Fm", "mul", ["Fs", "f12"])
    WS = np.zeros((3, 12, 1, 1), np.float32)
    for g in range(3):
        WS[g, g * 4:g * 4 + 4, 0, 0] = 1
    P["WS"] = WS
    N("SFC", "conv", ["Fm"], w="WS")
    N("supd", "slice", ["SFC"], start=0, stop=1)
    N("flrd", "slice", ["SFC"], start=1, stop=2)
    N("c11d", "slice", ["SFC"], start=2, stop=3)
    N("Fn", "slice", ["F"], start=14, stop=16)
    N("Fnm", "mul", ["Fn", "gVH"])
    P["W2s"] = np.ones((1, 2, 1, 1), np.float32)
    N("nr", "conv", ["Fnm"], w="W2s"); N("nrc", "min2", ["nr", "one"])
    N("Free", "slice", ["F"], start=12, stop=13)
    N("Hb0", "mul", ["Free", "c11d"]); N("Hb", "mul", ["Hb0", "nrc"])
    N("W0", "slice", ["F"], start=16, stop=17)
    N("f6", "vconcat", ["g4", "gVH"])

    WL = np.zeros((6, 2, 3, 3), np.float32)
    WL[0, 0, 0, 1] = 1; WL[1, 0, 2, 1] = 1; WL[2, 0, 1, 0] = 1; WL[3, 0, 1, 2] = 1
    WL[4, 1, 1, 0] = 1; WL[4, 1, 1, 2] = 1; WL[5, 1, 0, 1] = 1; WL[5, 1, 2, 1] = 1
    P["WL"] = WL
    P["W6s"] = np.ones((1, 6, 1, 1), np.float32)
    body = [
        {"id": "lA", "op": "mul", "in": ["W", "supd"]},
        {"id": "lWA", "op": "concat", "in": ["W", "lA"]},
        {"id": "lS", "op": "conv", "in": ["lWA"], "w": "WL"},
        {"id": "lSm", "op": "mul", "in": ["lS", "f6"]},
        {"id": "lin", "op": "conv", "in": ["lSm"], "w": "W6s"},
        {"id": "lnw", "op": "mul", "in": ["lin", "Free"]},
        {"id": "lW2", "op": "max2", "in": ["W", "lnw"]},
        {"id": "lWn", "op": "min2", "in": ["lW2", "one"]},
    ]
    G.append({"id": "H", "op": "loop", "T": 160,
              "state": [{"id": "W", "init": "W0"}], "body": body, "next": {"W": "lWn"}})

    N("wf", "mul", ["H.W", "flrd"]); N("fail", "gmax", ["wf"]); N("fwm", "gmean", ["wf"])
    N("tot", "gmean", ["Hb"]); N("hw", "mul", ["Hb", "H.W"]); N("fil", "gmean", ["hw"])
    N("sg9", "slice", ["F"], start=17, stop=18)
    N("sp", "mul", ["sg9", "coords"]); N("gsp", "gmean", ["sp"])
    N("sr", "vslice", ["gsp"], start=0, stop=1); N("sc_", "vslice", ["gsp"], start=1, stop=2)
    N("srh", "vmul", ["sr", "gH"]); N("scv", "vmul", ["sc_", "gV"]); N("s", "vadd", ["srh", "scv"])
    L("ok", [("tot", -40960.0), ("fil", 40960.0)], 5.0, "sig")
    L("nf", [("fail", -20.0)], 10.0, "sig")
    L("has", [("tot", 40960.0)], -5.0, "sig")
    N("s1", "vmul", ["ok", "nf"]); N("succ", "vmul", ["s1", "has"])
    L("sc", [("fil", 4096.0), ("fwm", -1024.0)])

    for k in range(7):
        N("m%d" % k, "vslice", ["mem"], start=k, stop=k + 1)
    for k in range(1, 7):
        N("a%d" % k, "vslice", ["aprev"], start=k, stop=k + 1)
    L("dpos", [("s", 1e5), ("m1", -1e5)], 0.0, "relu")
    L("dneg", [("s", -1e5), ("m1", 1e5)], 0.0, "relu")
    L("chg", [("dpos", 2.0), ("dneg", 2.0)], -10.0, "sig")
    L("mv", [("a1", 1.0), ("a2", 1.0), ("a3", 1.0), ("a4", 1.0)])
    L("nchg", [("chg", -1.0)], 1.0)
    N("blk", "vmul", ["mv", "nchg"])
    N("m0b", "vmul", ["m0", "blk"])
    L("m0n", [("m0", 1.0), ("blk", 1.0), ("m0b", -2.0)])
    L("m2s", [("m2", 1.0), ("blk", 1.0)])
    L("na6", [("a6", -1.0)], 1.0)
    N("m2n", "vmul", ["m2s", "na6"])
    N("m3a", "vmul", ["m3", "a6"])
    L("m3n", [("m3", 1.0), ("a6", 1.0), ("m3a", -2.0)])
    L("bet", [("sc", 1000.0), ("m4", -1000.0), ("a6", 2000.0),
              ("one", 1e7), ("m6", -1e7)], -100.0, "sig")
    L("dsc", [("sc", 1.0), ("m4", -1.0)])
    N("bd", "vmul", ["bet", "dsc"])
    L("m4n", [("m4", 1.0), ("bd", 1.0)])
    L("ds5", [("s", 1.0), ("m5", -1.0)])
    N("bd5", "vmul", ["bet", "ds5"])
    L("m5n", [("m5", 1.0), ("bd5", 1.0)])
    L("ds5n", [("s", 1.0), ("m5n", -1.0)])
    L("d5p", [("ds5n", 2e5)], 0.0, "relu")
    L("d5n", [("ds5n", -2e5)], 0.0, "relu")
    L("arr", [("d5p", -1.0), ("d5n", -1.0)], 10.0, "sig")
    L("dirR", [("ds5n", 2e5)], 0.0, "sig")
    L("ret", [("m2n", 10.0)], -15.0, "sig")
    N("clk", "vmul", ["ret", "arr"])
    L("dmr", [("dirR", 1.0), ("m0n", -1.0)])
    N("rdm", "vmul", ["ret", "dmr"])
    L("dfin", [("m0n", 1.0), ("rdm", 1.0)])
    L("pos", [("dfin", -1.0)], 1.0)
    N("tVp", "vmul", ["gV", "pos"]); N("tVn", "vmul", ["gV", "dfin"])
    N("tHp", "vmul", ["gH", "pos"]); N("tHn", "vmul", ["gH", "dfin"])
    L("lR", [("one", -5.0)])
    L("lA1", [("tHn", 2.0)]); L("lA2", [("tHp", 2.0)])
    L("lA3", [("tVn", 2.0)]); L("lA4", [("tVp", 2.0)])
    L("lA5", [("succ", 11.0)], -1.0)
    L("lA6", [("clk", 5.0)])
    L("lA7", [("one", -10.0)])
    N("logits", "vconcat", ["lR", "lA1", "lA2", "lA3", "lA4", "lA5", "lA6", "lA7"])
    N("mem_next", "vconcat", ["m0n", "s", "m2n", "m3n", "m4n", "m5n", "one"])

    P["Wr"] = np.array([1, 1, 0], np.float32).reshape(1, 3, 1, 1)
    N("rc", "conv", ["coords"], w="Wr")
    L("sgn", [("m3n", -0.01)], 0.005)
    N("rs", "mul", ["rc", "sgn"]); N("rs1", "add", ["rs", "one"])
    N("c8", "slice", ["F"], start=18, stop=19)
    N("click", "mul", ["c8", "rs1"])

    P = {k: np.asarray(v, np.float32) for k, v in P.items()}
    return {"graph": G, "memory_size": 7, "params": P}

import numpy as np

def build():
    P = {}; g = []
    def n(id_, op, ins, **kw):
        d = {"id": id_, "op": op, "in": ins}; d.update(kw); g.append(d)
    w = np.zeros((5, 3, 1, 1))
    w[0, :, 0, 0] = [63, 0, -1]; w[1, :, 0, 0] = [-189, 0, 160.5]
    w[2, :, 0, 0] = [-189, 0, 22.5]; w[3, :, 0, 0] = [189, 0, -157.5]
    w[4, :, 0, 0] = [-128 * 63, -2 * 63, 9000]
    P["WcR"] = w; P["Wneg"] = np.full((1, 1, 1, 1), -2.0); P["Bone"] = np.ones(1)
    P["Wone"] = np.ones((1, 1, 1, 1))
    w3 = np.zeros((3, 3, 1, 1))
    for i in range(3): w3[i, i, 0, 0] = -1
    P["Wm3"] = w3; P["Bm3"] = np.ones(3)
    n("cR", "conv", ["coords"], w="WcR"); n("s0", "slice", ["cR"], start=0, stop=1)
    n("ab0", "abs", ["s0"]); n("r1p", "conv", ["ab0"], w="Wneg", b="Bone")
    n("row1", "relu", ["r1p"]); n("s14", "slice", ["cR"], start=1, stop=4)
    n("r14", "relu", ["s14"]); n("m14", "conv", ["r14"], w="Wm3", b="Bm3")
    n("mk", "relu", ["m14"]); n("inv", "slice", ["mk"], start=0, stop=1)
    n("mA", "slice", ["mk"], start=1, stop=2); n("mB", "slice", ["mk"], start=2, stop=3)
    n("mid", "mul", ["mA", "mB"]); n("score", "slice", ["cR"], start=4, stop=5)
    nbc = [c for c in range(16) if c not in (0, 4)]
    f = np.zeros((6, 16, 7, 7)); fb = np.zeros(6)
    def put(ch, cols, dr, dc, v):
        for c in cols: f[ch, c, 3 + dr, 3 + dc] += v
    for ch in (0, 1):
        put(ch, nbc, 0, 0, 1); put(ch, nbc, 1, 3, 1); put(ch, nbc, -1, 0, -1); put(ch, nbc, 0, -1, -1)
    put(1, (0, 4), 1, 1, 1); fb[0] = -1; fb[1] = -2
    put(2, (2,), 1, 1, 1); put(2, (2,), 0, 1, -1); put(2, (2,), 1, 0, -1)
    put(3, nbc, 0, 0, 1); put(3, nbc, -1, 0, 1); put(3, nbc, 0, -1, -1); put(3, nbc, 0, 1, -1); fb[3] = -1
    put(4, [c for c in range(16) if c not in (4, 5)], 0, 0, 1); put(4, (4, 5), 0, -1, 1); fb[4] = -1
    put(5, (0,), 0, 0, 1)
    P["Wfa"] = f; P["Bfa"] = fb
    n("fa", "conv", ["x0"], w="Wfa", b="Bfa"); n("fr", "relu", ["fa"])
    for i, (nm, m) in enumerate([("A", "mid"), ("Hol", "mid"), ("Emp", "mid"),
                                 ("Bd", "mid"), ("Tm", "row1"), ("blk", "inv")]):
        n(nm + "_r", "slice", ["fr"], start=i, stop=i + 1); n(nm, "mul", [nm + "_r", m])
    n("Sd", "sub", ["A", "Hol"])
    te = np.zeros((16, 16, 3, 3))
    for c in range(16): te[c, c, 1, 1] = 1; te[c, c, 0, 1] = -1
    P["Wte"] = te
    n("tec", "conv", ["x0"], w="Wte"); n("ter", "relu", ["tec"]); n("tem", "mul", ["ter", "mid"])
    n("trs", "rowsum", ["tem"]); n("wv", "gmax", ["trs"])
    P["WL"] = np.eye(16) / 6.0; P["BL"] = np.full(16, -4.0 / 6.0)
    n("Lv", "dense", ["wv"], w="WL", b="BL"); n("HC", "mul", ["x0", "Hol"]); n("HL", "mul", ["HC", "Lv"])
    P["Wsum16"] = np.ones((1, 16, 1, 1))
    n("Ls", "conv", ["HL"], w="Wsum16"); n("Wt0", "add", ["Sd", "Emp"]); n("Wt", "add", ["Wt0", "Ls"])
    mkw = np.zeros((2, 1, 1, 1)); mkw[0, 0, 0, 0] = -1
    P["Wmk"] = mkw; P["Bmk"] = np.ones(2)
    n("Mk", "conv", ["Bd"], w="Wmk", b="Bmk"); n("Q1", "concat", ["Wt", "Tm"])
    ws = np.zeros((2, 2, 3, 3)); ws[0, 0, 1, 0] = 1; ws[1, 1, 1, 0] = 1
    P["Wsh"] = ws
    g.append({"id": "LP", "op": "loop", "T": 64, "state": [{"id": "p", "zeros": [2, 64, 64]}],
              "body": [{"id": "ps", "op": "conv", "in": ["p"], "w": "Wsh"},
                       {"id": "pm", "op": "mul", "in": ["ps", "Mk"]},
                       {"id": "pn", "op": "add", "in": ["pm", "Q1"]}], "next": {"p": "pn"}})
    n("Wp", "slice", ["LP.p"], start=0, stop=1); n("PT", "slice", ["LP.p"], start=1, stop=2)
    n("i1a", "sub", ["Wp", "Wt"]); n("idx1", "conv", ["i1a"], w="Wone", b="Bone")
    n("HI", "mul", ["HC", "idx1"]); n("st", "gmax", ["HI"])
    n("zb", "mul", ["Bd", "score"]); n("zbm", "gmax", ["zb"]); n("zbs", "sub", ["zb", "zbm"])
    n("fb0", "conv", ["zbs"], w="Wone", b="Bone"); n("fb1", "relu", ["fb0"]); n("FB", "mul", ["fb1", "Bd"])
    n("xFB", "mul", ["x0", "FB"]); n("mc", "gmax", ["xFB"]); n("vb", "vadd", ["st", "mc"])
    n("xi", "mul", ["x0", "vb"]); n("inj", "conv", ["xi"], w="Wsum16"); n("Bi", "mul", ["inj", "Bd"])
    n("R1", "slice", ["Mk"], start=0, stop=1)
    w1 = np.zeros((1, 1, 3, 3)); w1[0, 0, 1, 0] = 1; P["Wsh1"] = w1
    g.append({"id": "LQ", "op": "loop", "T": 64, "state": [{"id": "q", "zeros": [1, 64, 64]}],
              "body": [{"id": "qs", "op": "conv", "in": ["q"], "w": "Wsh1"},
                       {"id": "qm", "op": "mul", "in": ["qs", "R1"]},
                       {"id": "qn", "op": "add", "in": ["qm", "Bi"]}], "next": {"q": "qn"}})
    n("idx", "add", ["i1a", "LQ.q"])
    K = 14
    P["WkI"] = np.ones((K, 1, 1, 1)); P["BkI"] = -np.arange(1, K + 1, dtype=float)
    P["WdK"] = (-2.0 * np.eye(K)).reshape(K, K, 1, 1); P["BoneK"] = np.ones(K)
    P["WsumK"] = np.ones((1, K, 1, 1))
    for nm, src in (("I", "PT"), ("J", "idx")):
        n(nm + "a", "conv", [src], w="WkI", b="BkI"); n(nm + "b", "abs", [nm + "a"])
        n(nm + "c", "conv", [nm + "b"], w="WdK", b="BoneK"); n(nm + "d", "relu", [nm + "c"])
    n("I", "mul", ["Id", "Tm"])
    el = []
    for c in range(16):
        n(f"xc{c}", "slice", ["x0"], start=c, stop=c + 1); n(f"I{c}", "mul", ["I", f"xc{c}"])
        n(f"t{c}", "gmax", [f"I{c}"]); n(f"J{c}", "mul", ["Jd", f"t{c}"])
        n(f"E{c}", "conv", [f"J{c}"], w="WsumK"); el.append(f"E{c}")
    n("Ex", "concat", el); n("xE", "mul", ["x0", "Ex"]); n("match", "conv", ["xE"], w="Wsum16")
    n("Sm", "mul", ["Sd", "match"]); n("wr1", "add", ["Emp", "Sd"]); n("wrong", "sub", ["wr1", "Sm"])
    n("z", "mul", ["wrong", "score"]); n("zmx", "gmax", ["z"]); n("zm", "sub", ["z", "zmx"])
    n("fd0", "conv", ["zm"], w="Wone", b="Bone"); n("fd1", "relu", ["fd0"]); n("FW", "mul", ["fd1", "wrong"])
    n("SP", "mul", ["FW", "idx"]); n("Sg", "gmean", ["SP"])
    P["W4096"] = np.full((1, 1), 4096.0); n("Sv", "dense", ["Sg"], w="W4096")
    n("dd", "sub", ["PT", "Sv"]); n("ad", "abs", ["dd"]); n("mt0", "conv", ["ad"], w="Wneg", b="Bone")
    n("mt", "relu", ["mt0"]); n("Tsel", "mul", ["Tm", "mt"]); n("xT", "mul", ["x0", "Tsel"])
    n("cv", "gmax", ["xT"]); n("xm", "mul", ["x0", "cv"])
    wi = np.zeros((1, 16, 3, 3)); wi[0, :, 1, 1] = 1; wi[0, :, 2, 2] = 1; P["Winv"] = wi
    n("inv1", "conv", ["xm"], w="Winv"); P["Bm15"] = np.full(1, -1.5)
    n("inv2", "conv", ["inv1"], w="Wone", b="Bm15"); n("inv3", "relu", ["inv2"])
    n("invmap", "mul", ["inv3", "inv"])
    wc = np.zeros((1, 16, 3, 3)); wc[0, 0, 1, 1] = 1; wc[0, 0, 0, 1] = -1; wc[0, 0, 1, 0] = -1
    P["Wcor"] = wc
    n("cor0", "conv", ["x0"], w="Wcor"); n("cor1", "relu", ["cor0"]); n("cor", "mul", ["cor1", "inv"])
    s16 = np.zeros((16, 16, 3, 3))
    for c in range(16): s16[c, c, 2, 2] = 1
    P["Wsh16"] = s16
    n("xsh", "conv", ["x0"], w="Wsh16"); n("xsc", "mul", ["xsh", "cor"]); n("selcol", "gmax", ["xsc"])
    n("scv", "vmul", ["selcol", "cv"]); P["Wsum16v"] = np.ones((1, 16))
    n("selok", "dense", ["scv"], w="Wsum16v"); n("sel", "gmax", ["blk"]); n("good", "vmul", ["sel", "selok"])
    P["Wns"] = np.full((1, 1), -1.0); n("ngood", "dense", ["good"], w="Wns", b="Bone")
    n("c1", "mul", ["FW", "good"]); n("c2", "mul", ["invmap", "ngood"]); n("click", "add", ["c1", "c2"])
    n("cfl", "gmax", ["click"]); n("wf", "gmax", ["wrong"]); n("lv", "vconcat", ["cfl", "wf"])
    wl = np.zeros((8, 2)); wl[6, 0] = 4.0; wl[7, 1] = 1.0
    bl = np.full(8, -10.0); bl[5] = 0.5; bl[6] = 0.0; bl[7] = 0.0
    P["Wl"] = wl; P["Bl"] = bl
    n("logits", "dense", ["lv"], w="Wl", b="Bl")
    params = {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 0, "params": params}

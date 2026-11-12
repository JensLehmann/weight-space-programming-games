import numpy as np

def build():
    P = {}
    g = []
    def N(i, op, ins, **kw):
        d = {"id": i, "op": op, "in": ins}
        d.update(kw)
        g.append(d)
    f32 = np.float32
    Wf = np.zeros((1, 16, 3, 3), f32)
    Wf[0, 5, 1, 1] = 1; Wf[0, 10, 1, 1] = 1; Wf[0, 9, :, :] = -1
    P["Wf"] = Wf
    N("fr", "conv", ["x0"], w="Wf"); N("Fraw", "relu", ["fr"])
    Wrm = np.zeros((4, 3, 1, 1), f32); brm = np.array([0, -1, 126, 125], f32)
    Wrm[0, 0] = 126; Wrm[1, 0] = 126; Wrm[2, 0] = -126; Wrm[3, 0] = -126
    P["Wrm"] = Wrm; P["brm"] = brm
    N("rm1", "conv", ["coords"], w="Wrm", b="brm"); N("rm2", "relu", ["rm1"])
    P["Wrm2"] = np.array([1, -1, 1, -1], f32).reshape(1, 4, 1, 1); P["brm2"] = np.array([-1], f32)
    N("rm3", "conv", ["rm2"], w="Wrm2", b="brm2"); N("rmask", "relu", ["rm3"])
    N("F", "mul", ["Fraw", "rmask"])
    Wf2 = np.zeros((1, 16, 1, 1), f32)
    for c in (5, 9, 10, 12, 14, 15):
        Wf2[0, c] = 1
    P["Wf2"] = Wf2
    N("f2r", "conv", ["x0"], w="Wf2"); N("F2", "mul", ["f2r", "rmask"])
    Wgeo = np.zeros((4, 3, 1, 1), f32)
    Wgeo[0, 1] = 63; Wgeo[1, 0] = 63; Wgeo[2, 1] = -63; Wgeo[3, 0] = -63
    P["Wgeo"] = Wgeo; P["bgeo"] = np.array([0, 0, 63, 63], f32)
    N("geo", "conv", ["coords"], w="Wgeo", b="bgeo")
    N("Pc", "slice", ["x0"], start=10, stop=11)
    N("PG", "mul", ["Pc", "geo"]); N("gm", "gmax", ["PG"])
    WdA = np.array([[2, 0, -2, 0], [0, 2, 0, -2], [1, 0, 1, 0], [1, 0, 1, 0]], f32)
    P["WdA"] = WdA; P["bdA"] = np.array([126, 126, -68.5, -69.5], f32)
    N("dA", "dense", ["gm"], w="WdA", b="bdA"); N("hA", "vrelu", ["dA"])
    WdB = np.zeros((6, 4), f32)
    WdB[0, 0] = 1; WdB[1, 1] = 1; WdB[2, 0] = 1; WdB[3, 1] = 1
    WdB[4, 2] = 1; WdB[4, 3] = -1; WdB[5, 2] = -1; WdB[5, 3] = 1
    P["WdB"] = WdB; P["bdB"] = np.array([0, 0, 0, 0, 0, 1], f32)
    N("dB", "dense", ["hA"], w="WdB", b="bdB")
    N("mids4", "vslice", ["dB"], start=0, stop=4); N("sv", "vslice", ["dB"], start=4, stop=6)
    Wgn = np.zeros((4, 3, 1, 1), f32)
    Wgn[0, 1] = -252; Wgn[1, 0] = -252; Wgn[2, 1] = -252; Wgn[3, 0] = -252
    P["Wgn"] = Wgn; P["bgn"] = np.array([0, 0, -1, -1], f32)
    N("gn", "conv", ["coords"], w="Wgn", b="bgn")
    N("Ka", "add", ["gn", "mids4"]); N("K", "relu", ["Ka"])
    N("K1", "slice", ["K"], start=0, stop=2); N("K2", "slice", ["K"], start=2, stop=4)
    N("st", "sub", ["K1", "K2"]); N("LL", "mul", ["st", "Pc"]); N("Lw", "mul", ["LL", "sv"])
    N("cat3", "concat", ["Lw", "Pc"])
    WS = np.zeros((4, 3, 1, 1), f32)
    WS[0, :, 0, 0] = [1, 1, 0]; WS[1, :, 0, 0] = [-1, -1, 1]
    WS[2, :, 0, 0] = [1000, 1000, 0]; WS[3, :, 0, 0] = [-1000, -1000, 1000]
    P["WS"] = WS
    N("LRS", "conv", ["cat3"], w="WS")
    N("LR", "slice", ["LRS"], start=0, stop=2); N("seeds", "slice", ["LRS"], start=2, stop=4)
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]
    Wsh = np.zeros((16, 2, 5, 5), f32); bsh = np.zeros(16, f32)
    for gi, (di, dj) in enumerate(offs):
        for c in range(2):
            o = 2 * gi + c
            Wsh[o, c, 2 + di, 2 + dj] = 1; bsh[o] = -(abs(di) + abs(dj))
    P["Wsh"] = Wsh; P["bsh"] = bsh
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    Wsc = np.zeros((20, 5, 3, 3), f32)
    for d, (di, dj) in enumerate(dirs):
        for ch in range(5):
            Wsc[5 * d + ch, ch, 1 + di, 1 + dj] = 1
    P["Wsc"] = Wsc
    D = np.zeros((6, 8), f32)
    D[0, [0, 1]] = 1; D[1, [2, 3]] = 1; D[2, [4, 7]] = 1; D[3, [6, 5]] = 1
    D[4] = D[3] - D[2]; D[5] = D[2] - D[3]
    P["Dsc"] = (D * 4096 * 1.3).astype(f32)
    P["Dsc2"] = (D * 4096).astype(f32)

    def field_score(s, Fn, Dn, T):
        body = [{"id": "Lcs" + s, "op": "conv", "in": ["V" + s], "w": "Wsh", "b": "bsh"}]
        for gi in range(8):
            body.append({"id": "Ls%d%s" % (gi, s), "op": "slice", "in": ["Lcs" + s],
                         "start": 2 * gi, "stop": 2 * gi + 2})
        prev = "Ls0" + s
        for gi in range(1, 8):
            nid = "Lm%d%s" % (gi, s)
            body.append({"id": nid, "op": "max2", "in": [prev, "Ls%d%s" % (gi, s)]}); prev = nid
        body += [{"id": "Lr" + s, "op": "relu", "in": [prev]},
                 {"id": "Lf" + s, "op": "mul", "in": ["Lr" + s, Fn]},
                 {"id": "Ln" + s, "op": "max2", "in": ["Lf" + s, "seeds"]}]
        H = "H" + s
        g.append({"id": H, "op": "loop", "T": T, "state": [{"id": "V" + s, "init": "seeds"}],
                  "body": body, "next": {"V" + s: "Ln" + s}})
        N("VL" + s, "slice", [H + ".V" + s], start=0, stop=1)
        N("VR" + s, "slice", [H + ".V" + s], start=1, stop=2)
        N("X" + s, "concat", ["LR", "VR" + s, "VL" + s, Fn])
        N("O" + s, "slice", ["X" + s], start=2, stop=4)
        N("SH" + s, "conv", ["X" + s], w="Wsc")
        gvs = []
        for d in range(4):
            t = "%d%s" % (d, s)
            N("so" + t, "slice", ["SH" + s], start=5 * d, stop=5 * d + 2)
            N("sO" + t, "slice", ["SH" + s], start=5 * d + 2, stop=5 * d + 4)
            N("sF" + t, "slice", ["SH" + s], start=5 * d + 4, stop=5 * d + 5)
            N("oo" + t, "mul", ["LR", "so" + t]); N("e" + t, "sub", ["LR", "oo" + t])
            N("gg" + t, "sub", ["sO" + t, "O" + s]); N("t1" + t, "mul", ["e" + t, "gg" + t])
            N("t2" + t, "mul", ["t1" + t, "sF" + t]); N("gv" + t, "gmean", ["t2" + t])
            gvs.append("gv" + t)
        N("vec8" + s, "vconcat", gvs)
        N("sc6" + s, "dense", ["vec8" + s], w=Dn)
        return "sc6" + s

    a1 = field_score("", "F", "Dsc", 190)
    a2 = field_score("b", "F2", "Dsc2", 190)
    P["Wab"] = np.concatenate([np.eye(6), -np.eye(6)], 0).astype(f32)
    N("ab", "dense", [a1], w="Wab"); N("abr", "vrelu", ["ab"])
    P["Wis"] = np.full((1, 12), -10, f32); P["bis"] = np.array([1], f32)
    N("inds", "dense", ["abr"], w="Wis", b="bis"); N("ind", "vrelu", ["inds"])
    N("a2g", "vmul", [a2, "ind"])
    N("sc6t", "vadd", [a1, "a2g"])
    N("mem01", "vslice", ["mem"], start=0, stop=2)
    N("gmm", "vconcat", ["gm", "mem01"])
    Wm1 = np.zeros((4, 6), f32)
    Wm1[0] = [1, 0, 1, 0, 0, -1]; Wm1[1] = Wm1[0]; Wm1[2] = -Wm1[0]; Wm1[3] = -Wm1[0]
    P["Wm1"] = Wm1; P["bm1"] = np.array([-64, -65, 62, 61], f32)
    N("dm", "dense", ["gmm"], w="Wm1", b="bm1"); N("hm", "vrelu", ["dm"])
    P["Wm2"] = np.array([[1, -1, 0, 0], [0, 0, 1, -1]], f32)
    N("pn", "dense", ["hm"], w="Wm2")
    Wa = np.zeros((4, 8), f32); Wa[0, 4] = 1; Wa[1, 3] = 1; Wa[2, 3] = 1; Wa[3, 4] = 1
    P["Wa"] = Wa
    N("aa", "dense", ["aprev"], w="Wa")
    N("x1", "vslice", ["aa"], start=0, stop=2); N("x2", "vslice", ["aa"], start=2, stop=4)
    N("p1", "vmul", ["pn", "x1"]); N("p2", "vmul", ["pn", "x2"]); N("pp", "vconcat", ["p1", "p2"])
    P["We"] = np.array([[1, 1, 0, 0], [-1, -1, -1, -1]], f32); P["be"] = np.array([0, 1], f32)
    N("ee", "dense", ["pp"], w="We", b="be")
    N("mv", "vslice", ["mem"], start=0, stop=1); N("keep", "vslice", ["ee"], start=1, stop=2)
    N("e2v", "vslice", ["ee"], start=0, stop=1); N("mk", "vmul", ["mv", "keep"])
    N("mnew", "vadd", ["mk", "e2v"])
    P["Wg"] = np.array([[1, 0, 1, 0]], f32); P["bg"] = np.array([-63], f32)
    N("gnow", "dense", ["gm"], w="Wg", b="bg")
    N("PCo", "mul", ["Pc", "coords"]); N("cg", "gmean", ["PCo"])
    N("sig", "vconcat", ["gm", "cg"])
    N("psig", "vslice", ["mem"], start=2, stop=9)
    N("sp", "vconcat", ["sig", "psig"])
    sc = np.array([1, 1, 1, 1, 4096 * 63, 4096 * 63, 4096], f32)
    Wdf = np.zeros((14, 14), f32)
    for i in range(7):
        Wdf[i, i] = sc[i]; Wdf[i, 7 + i] = -sc[i]
        Wdf[7 + i, i] = -sc[i]; Wdf[7 + i, 7 + i] = sc[i]
    P["Wdf"] = Wdf
    N("dfl", "dense", ["sp"], w="Wdf"); N("dfr", "vrelu", ["dfl"])
    P["Wnc"] = -np.ones((1, 14), f32); P["bnc"] = np.array([1], f32)
    N("ncl", "dense", ["dfr"], w="Wnc", b="bnc"); N("nc", "vrelu", ["ncl"])
    N("apv", "vslice", ["aprev"], start=1, stop=5)
    N("bp", "vslice", ["mem"], start=9, stop=13)
    N("bprod", "vmul", ["bp", "apv"])
    N("bcat", "vconcat", ["bp", "apv", "bprod"])
    Wu = np.zeros((4, 12), f32)
    for i in range(4):
        Wu[i, i] = 1; Wu[i, 4 + i] = 1; Wu[i, 8 + i] = -1
    P["Wu"] = Wu
    N("bu", "dense", ["bcat"], w="Wu")
    N("bnext", "vmul", ["bu", "nc"])
    N("mem_next", "vconcat", ["mnew", "gnow", "sig", "bnext"])
    N("base4", "vslice", ["sc6t"], start=0, stop=4); N("dif2", "vslice", ["sc6t"], start=4, stop=6)
    N("md", "vmul", ["dif2", "mnew"])
    N("lin", "vconcat", ["base4", "md", "aprev", "bnext"])
    Wl = np.zeros((8, 18), f32)
    Wl[1, 0] = 1; Wl[2, 1] = 1; Wl[3, 2] = 1; Wl[3, 4] = 1; Wl[4, 3] = 1; Wl[4, 5] = 1
    for a in range(1, 5):
        Wl[a, 6 + a] += 0.5
        Wl[a, 14 + a - 1] = -3
    Wl[1, 6 + 2] = -1; Wl[2, 6 + 1] = -1; Wl[3, 6 + 4] = -1; Wl[4, 6 + 3] = -1
    P["Wl"] = Wl; P["bl"] = np.array([-1000, 0, 0, 0, 0, -1000, -1000, -1000], f32)
    N("logits", "dense", ["lin"], w="Wl", b="bl")
    P["Wc"] = np.zeros((1, 1, 1, 1), f32)
    N("click", "conv", ["F"], w="Wc")
    return {"graph": g, "memory_size": 13, "params": P}

import numpy as np

def build():
    P = {}
    g = []
    def n(id_, op, ins, **kw):
        d = {"id": id_, "op": op, "in": ins}; d.update(kw); g.append(d)
    def vclamp(id_, src, size, scale):
        P["Wc1_" + id_] = -scale * np.eye(size); P["Wc2_" + id_] = -np.eye(size)
        P["bc_" + id_] = np.ones(size)
        n(id_ + "_a", "vrelu", [src])
        n(id_ + "_b", "dense", [id_ + "_a"], w="Wc1_" + id_, b="bc_" + id_)
        n(id_ + "_c", "vrelu", [id_ + "_b"])
        n(id_, "dense", [id_ + "_c"], w="Wc2_" + id_, b="bc_" + id_)

    Wp = np.zeros((1, 16, 1, 1)); Wp[0, 10] = 1; Wp[0, 1] = 1; P["Wp"] = Wp
    n("Pm", "conv", ["x0"], w="Wp")
    n("rowc", "slice", ["coords"], start=0, stop=1)
    n("col", "slice", ["coords"], start=1, stop=2)
    n("one", "slice", ["coords"], start=2, stop=3)
    n("prm", "rowmax", ["Pm"]); n("pcm", "colmax", ["Pm"])
    n("gr", "gmean", ["prm"]); n("gc", "gmean", ["pcm"])
    n("grc", "vconcat", ["gr", "gc"])
    P["Wrc"] = np.array([[64.0, -64.0]])
    n("drc", "dense", ["grc"], w="Wrc")
    vclamp("vflag", "drc", 1, 1.0)
    n("rmc", "sub", ["rowc", "col"]); n("vk", "mul", ["rmc", "vflag"])
    n("key", "add", ["col", "vk"]); n("nkey", "sub", ["one", "key"])
    n("pk", "mul", ["Pm", "key"]); n("maxk", "gmax", ["pk"])
    n("pnk", "mul", ["Pm", "nkey"]); n("gnk", "gmax", ["pnk"])
    n("mgk", "vconcat", ["maxk", "gnk"])
    P["Wmid"] = np.array([[0.5, -0.5]]); P["bmid"] = np.array([0.5])
    n("mid", "dense", ["mgk"], w="Wmid", b="bmid")
    n("pc", "mul", ["Pm", "col"]); n("maxc", "gmax", ["pc"])
    Wnc = np.zeros((1, 3, 1, 1)); Wnc[0, 1] = -1; Wnc[0, 2] = 1; P["Wnc"] = Wnc
    n("ncol", "conv", ["coords"], w="Wnc")
    n("pn", "mul", ["Pm", "ncol"]); n("gn", "gmax", ["pn"])
    n("mg", "vconcat", ["maxc", "gn"])
    P["Wsp"] = np.array([[1.0, 1.0]]); P["bsp"] = np.array([-1.0])
    n("spread", "dense", ["mg"], w="Wsp", b="bsp")
    n("dcol", "sub", ["mid", "key"])
    P["Wt"] = np.array([126.0, -126.0]).reshape(2, 1, 1, 1)
    n("tt", "conv", ["dcol"], w="Wt"); n("rt", "relu", ["tt"])
    n("ct", "min2", ["rt", "one"]); n("LR", "mul", ["ct", "Pm"])

    Wrm = np.zeros((2, 3, 1, 1)); Wrm[0, 0] = 63; Wrm[1, 0] = -63; Wrm[1, 2] = 63
    P["Wrm"] = Wrm
    n("rmk", "conv", ["coords"], w="Wrm")
    n("rmA", "slice", ["rmk"], start=0, stop=1); n("rmB", "slice", ["rmk"], start=1, stop=2)
    n("rmC", "min2", ["rmA", "rmB"]); n("rmask", "min2", ["rmC", "one"])
    Wfl = np.zeros((2, 16, 3, 3))
    for c in (5, 10, 1):
        Wfl[0, c, 1, 1] = 1
    Wfl[1, 5, 1, 1] = 1
    Wfl[:, 9] = -3
    P["Wfl"] = Wfl
    n("fl0", "conv", ["x0"], w="Wfl"); n("fl1", "relu", ["fl0"])
    n("flA", "min2", ["fl1", "one"]); n("flAm", "mul", ["flA", "rmask"])
    n("fl", "slice", ["flAm"], start=0, stop=1)
    n("flN", "slice", ["flAm"], start=1, stop=2)
    W9 = np.zeros((1, 16, 3, 3)); W9[0, 9] = 1; P["W9"] = W9
    n("n9", "conv", ["x0"], w="W9"); n("dil9", "min2", ["n9", "one"])
    n("flB", "max2", ["fl", "dil9"])
    n("flcat", "concat", ["fl", "fl", "flB", "flB"])
    Wsrc = np.zeros((4, 2, 1, 1))
    for c in range(4):
        Wsrc[c, c % 2] = 1000
    P["Wsrc"] = Wsrc
    n("S", "conv", ["LR"], w="Wsrc")

    Wsh = np.zeros((16, 4, 3, 3))
    for k, (di, dj) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        for c in range(4):
            Wsh[4 * k + c, c, 1 + di, 1 + dj] = 1
    P["Wsh"] = Wsh; P["bsh"] = -np.ones(16)
    body = [
        {"id": "sh", "op": "conv", "in": ["F"], "w": "Wsh", "b": "bsh"},
        {"id": "s0", "op": "slice", "in": ["sh"], "start": 0, "stop": 4},
        {"id": "s1", "op": "slice", "in": ["sh"], "start": 4, "stop": 8},
        {"id": "s2", "op": "slice", "in": ["sh"], "start": 8, "stop": 12},
        {"id": "s3", "op": "slice", "in": ["sh"], "start": 12, "stop": 16},
        {"id": "m1", "op": "max2", "in": ["s0", "s1"]},
        {"id": "m2", "op": "max2", "in": ["s2", "s3"]},
        {"id": "mm", "op": "max2", "in": ["m1", "m2"]},
        {"id": "mr", "op": "relu", "in": ["mm"]},
        {"id": "rf", "op": "mul", "in": ["mr", "flcat"]},
        {"id": "Fn", "op": "max2", "in": ["rf", "S"]},
    ]
    g.append({"id": "H", "op": "loop", "T": 256,
              "state": [{"id": "F", "zeros": [4, 64, 64]}],
              "body": body, "next": {"F": "Fn"}})
    n("FLf", "slice", ["H.F"], start=0, stop=1); n("FRf", "slice", ["H.F"], start=1, stop=2)
    n("FLb", "slice", ["H.F"], start=2, stop=3); n("FRb", "slice", ["H.F"], start=3, stop=4)
    n("FX", "concat", ["FRf", "FLf"]); n("FXB", "concat", ["FRb", "FLb"])

    Wmv = np.zeros((8, 2, 3, 3))
    for k, (a, b) in enumerate([(2, 1), (0, 1), (1, 2), (1, 0)]):
        for c in range(2):
            Wmv[2 * k + c, c, a, b] = 1
    P["Wmv"] = Wmv
    n("shb", "conv", ["LR"], w="Wmv")
    n("blob8", "concat", ["LR", "LR", "LR", "LR"])
    n("frd", "sub", ["shb", "blob8"]); n("fr", "relu", ["frd"])
    Wd = np.zeros((8, 10))
    for k in range(8):
        Wd[k, k] = 1; Wd[k, 8 + k % 2] = -1
    P["Wd"] = Wd
    Wg = np.zeros((8, 16))
    for k in range(8):
        Wg[k, k] = 1; Wg[k, 8 + k] = 3.5
    P["Wg"] = Wg
    P["Wu"] = -np.eye(8); P["bu1"] = np.ones(8)
    for t, fx in (("T", "FX"), ("B", "FXB")):
        n("FX8" + t, "concat", [fx, fx, fx, fx])
        n("vv" + t, "mul", ["fr", "FX8" + t]); n("V" + t, "gmax", ["vv" + t])
        n("cur" + t, "mul", ["LR", fx]); n("C" + t, "gmax", ["cur" + t])
        n("VC" + t, "vconcat", ["V" + t, "C" + t])
        n("dv" + t, "dense", ["VC" + t], w="Wd"); n("rdv" + t, "vrelu", ["dv" + t])
        n("drd" + t, "vconcat", ["dv" + t, "rdv" + t])
        n("g0" + t, "dense", ["drd" + t], w="Wg")
        n("u" + t, "dense", ["V" + t], w="Wu", b="bu1"); n("ru" + t, "vrelu", ["u" + t])
        n("ind" + t, "dense", ["ru" + t], w="Wu", b="bu1")
        n("gain" + t, "vmul", ["g0" + t, "ind" + t])
    n("CL", "vslice", ["CT"], start=0, stop=1)
    vclamp("conn", "CL", 1, 1.0)
    n("gdT", "dense", ["gainT"], w="Wu")
    n("gdiff", "vadd", ["gainB", "gdT"])
    n("gdc", "vmul", ["gdiff", "conn"])
    n("ngdc", "dense", ["gdc"], w="Wu")
    n("gain", "vadd", ["gainB", "ngdc"])

    n("x1", "slice", ["x0"], start=1, stop=2); n("sel", "gmax", ["x1"])
    P["Wpu"] = np.array([[0, 1.0, 1, 1, 1, 0, 0, 0]])
    n("pushed", "dense", ["aprev"], w="Wpu")
    n("selp", "vmul", ["sel", "pushed"])
    P["Wneg"] = np.array([[-1.0]]); P["bpos"] = np.array([1.0])
    n("npush", "dense", ["pushed"], w="Wneg", b="bpos")
    n("selnp", "vmul", ["sel", "npush"])
    n("nsel", "dense", ["sel"], w="Wneg", b="bpos")
    n("nconn", "dense", ["conn"], w="Wneg", b="bpos")

    n("frb", "mul", ["fr", "dil9"]); n("BK", "gmax", ["frb"])
    n("rgB", "vrelu", ["gainB"]); n("wk", "vmul", ["BK", "rgB"])
    W4 = np.zeros((4, 8))
    for d in range(4):
        W4[d, 2 * d] = 1; W4[d, 2 * d + 1] = 1
    P["W4"] = W4
    n("want4", "dense", ["wk"], w="W4")
    vclamp("want", "want4", 4, 1000.0)
    P["Wsum4"] = np.ones((1, 4))
    n("wany", "dense", ["want"], w="Wsum4")
    vclamp("wanyc", "wany", 1, 1.0)
    n("stold", "vslice", ["mem"], start=6, stop=7)
    n("nstold", "dense", ["stold"], w="Wneg", b="bpos")
    n("tr1", "vmul", ["wanyc", "nconn"]); n("tr2", "vmul", ["tr1", "nsel"])
    n("trig", "vmul", ["tr2", "nstold"])

    n("rold", "vslice", ["mem"], start=0, stop=1); n("spold", "vslice", ["mem"], start=1, stop=2)
    n("dold", "vslice", ["mem"], start=2, stop=6)
    n("spp", "vconcat", ["spread", "spold"])
    P["Wds"] = np.array([[1.0, -1.0]])
    n("dsp", "dense", ["spp"], w="Wds")
    Wad = np.zeros((1, 8)); Wad[0, 3] = 1; Wad[0, 4] = -1; P["Wad"] = Wad
    n("adir", "dense", ["aprev"], w="Wad")
    n("ev", "vmul", ["dsp", "adir"]); n("re", "vconcat", ["rold", "ev"])
    P["Wr"] = np.array([[1.0, 200.0]])
    n("rnew", "dense", ["re"], w="Wr"); n("sb", "vtanh", ["rnew"])
    P["Wm4"] = -np.eye(4)
    n("ndold", "dense", ["dold"], w="Wm4")
    n("dd", "vadd", ["want", "ndold"]); n("ddt", "vmul", ["dd", "trig"])
    n("dnew", "vadd", ["dold", "ddt"])

    Wbs = np.zeros((1, 16, 5, 5))
    Wbs[0, 11, 2, 2] = 1; Wbs[0, 5, 2, 1] = 1; Wbs[0, 5, 2, 4] = 1
    Wbs[0, 5, 1, 2] = 1; Wbs[0, 5, 4, 2] = 1
    P["Wbs"] = Wbs; P["bbs"] = np.array([-4.0])
    n("bs0", "conv", ["x0"], w="Wbs", b="bbs"); n("Bsel", "relu", ["bs0"])
    Wnb = np.zeros((4, 1, 7, 7))
    for d, (di, dj) in enumerate([(-3, 0), (3, 0), (0, -3), (0, 3)]):
        Wnb[d, 0, 3 + di, 3 + dj] = 1
    P["Wnb"] = Wnb
    n("nbf", "conv", ["flN"], w="Wnb"); n("cmv", "mul", ["nbf", "Bsel"])
    n("CM", "gmax", ["cmv"])
    n("cms", "dense", ["CM"], w="Wsum4")
    vclamp("cmany", "cms", 1, 1000.0)
    n("ncm", "dense", ["cmany"], w="Wneg", b="bpos")
    n("stuck", "vmul", ["sel", "ncm"])
    n("stsum", "vadd", ["stold", "stuck"])
    vclamp("stnew", "stsum", 1, 1.0)
    n("dw", "vmul", ["CM", "dnew"])
    n("dcm", "vconcat", ["CM", "dw"])
    Wsc = np.zeros((4, 8))
    for d in range(4):
        Wsc[d, d] = 1; Wsc[d, 4 + d] = 2
    P["Wsc"] = Wsc
    n("sc4", "dense", ["dcm"], w="Wsc")
    n("selsc", "vmul", ["sc4", "selnp"])

    n("msrc", "vconcat", ["sb", "spread", "dnew", "stnew"])
    Wmn = np.eye(7); Wmn[0, 0] = 2.0; P["Wmn"] = Wmn
    n("mem_next", "dense", ["msrc"], w="Wmn")

    Wdf = np.zeros((1, 8)); Wdf[0, 4] = 1; Wdf[0, 7] = 1; Wdf[0, 5] = -1; Wdf[0, 6] = -1
    P["Wdf"] = Wdf
    n("diff", "dense", ["gain"], w="Wdf"); n("sdiff", "vmul", ["sb", "diff"])
    n("ssq", "vmul", ["sb", "sb"])
    # feat: gain 0-7, sdiff 8, ssq 9, aprev 10-17, selp 18, trig 19, selsc 20-23, stuck 24
    n("feat", "vconcat", ["gain", "sdiff", "ssq", "aprev", "selp", "trig", "selsc", "stuck"])
    Wo = np.zeros((8, 25)); bo = np.zeros(8)
    bo[0] = bo[5] = bo[7] = -1e6
    Wo[1, 0] = Wo[1, 1] = 1; Wo[1, 12] = -0.6
    Wo[2, 2] = Wo[2, 3] = 1; Wo[2, 11] = -0.6
    for k in range(4, 8):
        Wo[3, k] = 0.5; Wo[4, k] = 0.5
    Wo[3, 8] = 0.5; Wo[4, 8] = -0.5
    Wo[3, 9] = -0.25; bo[3] = 0.25
    Wo[3, 14] = -0.6; Wo[4, 13] = -0.6
    for d in range(4):
        Wo[1 + d, 20 + d] = 1e4
    Wo[6, 18] = 1e5; Wo[6, 19] = 1e5; Wo[6, 24] = 1e5; bo[6] = -5e4
    P["Wo"] = Wo; P["bo"] = bo
    n("logits", "dense", ["feat"], w="Wo", b="bo")

    P["Wbox"] = np.ones((1, 1, 7, 7))
    n("near", "conv", ["Pm"], w="Wbox")
    n("x9", "slice", ["x0"], start=9, stop=10)
    n("nr1", "add", ["near", "one"]); n("c9", "mul", ["x9", "nr1"])
    P["Wk1"] = np.full((1, 1, 1, 1), 1000.0)
    n("c1", "conv", ["x1"], w="Wk1")
    n("click", "add", ["c1", "c9"])

    params = {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 7, "params": params}

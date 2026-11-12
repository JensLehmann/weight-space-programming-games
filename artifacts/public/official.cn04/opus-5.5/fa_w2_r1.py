import numpy as np

def build():
    P = {}
    g = []
    def n(i, op, ins, **kw):
        d = {"id": i, "op": op, "in": ins}
        d.update(kw)
        g.append(d)

    WQ = np.zeros((5, 3, 1, 1), np.float32)
    WQ[0, :, 0, 0] = [63, 0, 0]
    WQ[1, :, 0, 0] = [0, 1, 0]
    WQ[2, :, 0, 0] = [0, -1, 1]
    WQ[3, :, 0, 0] = [1, 0, 0]
    WQ[4, :, 0, 0] = [-1, 0, 1]
    P["WQ"] = WQ
    P["Wadj"] = np.ones((1, 1, 7, 7), np.float32)
    WRC = np.zeros((128, 3, 1, 1), np.float32)
    for i in range(64):
        WRC[i, :, 0, 0] = [63, 0, -i]
        WRC[64 + i, :, 0, 0] = [0, 63, -i]
    P["WRC"] = WRC
    P["WD"] = (-2.0 * np.eye(128, dtype=np.float32)).reshape(128, 128, 1, 1)
    P["bD"] = np.full(128, 1.5, np.float32)
    P["Wnf"] = np.array([[-1.0]], np.float32)
    P["bnf"] = np.array([1.0], np.float32)
    P["Wneg"] = np.array([[-1.0]], np.float32)
    P["Wnc"] = np.array([[-1.0, 1.0, -1.0]], np.float32)
    P["bnc"] = np.array([1.0], np.float32)
    WB = np.zeros((2, 128, 1, 1), np.float32)
    WB[0, :64] = 1
    WB[1, 64:] = 1
    P["WB"] = WB

    thr = 1.5 / 63.0
    W1 = np.zeros((6, 8), np.float32)
    dx = np.array([-.5, .5, 0, 0, .5, -.5, 0, 0], np.float32)
    dy = np.array([0, 0, -.5, .5, 0, 0, .5, -.5], np.float32)
    W1[0] = -dy; W1[1] = dy; W1[2] = -dx; W1[3] = dx
    W1[4] = [1, 1, -1, -1, 0, 0, 0, 0]
    W1[5] = [0, 0, 0, 0, 1, 1, -1, -1]
    P["W1"] = W1
    P["b1"] = np.array([-thr, -thr, -thr, -thr, 0, 0], np.float32)
    P["Wm"] = np.array([[-1000.0]], np.float32)
    Wout = np.zeros((8, 6), np.float32)
    for k in range(5):
        Wout[1 + k, k] = 1.0
    Wout[6, 5] = 3000.0
    P["Wout"] = Wout
    P["bout"] = np.array([-10, 0, 0, 0, 0, 0.001, -1500, -10], np.float32)

    n("blk", "slice", ["x0"], start=0, stop=1)
    n("x4", "slice", ["x0"], start=4, stop=5)
    n("e8", "slice", ["x0"], start=8, stop=9)
    n("g3", "slice", ["x0"], start=3, stop=4)
    n("Q", "conv", ["coords"], w="WQ")
    n("r63", "slice", ["Q"], start=0, stop=1)
    n("col", "slice", ["Q"], start=1, stop=2)
    n("ncol", "slice", ["Q"], start=2, stop=3)
    n("row", "slice", ["Q"], start=3, stop=4)
    n("nrow", "slice", ["Q"], start=4, stop=5)
    n("one", "slice", ["coords"], start=2, stop=3)
    n("rm", "min2", ["r63", "one"])
    n("blkc", "mul", ["blk", "rm"])
    n("x4m", "mul", ["x4", "rm"])
    n("grey", "gmax", ["x4m"])
    n("ng", "dense", ["grey"], w="Wnf", b="bnf")
    n("hasblk", "gmax", ["blkc"])
    n("hgp", "max2", ["blkc", "x4m"])
    n("hg", "gmax", ["hgp"])
    n("has8", "gmax", ["e8"])
    n("adj", "conv", ["blkc"], w="Wadj")
    n("adjb", "min2", ["adj", "one"])
    n("s8", "mul", ["e8", "adjb"])
    n("T8", "sub", ["e8", "s8"])
    n("t_c", "mul", ["T8", "col"])
    n("t_nc", "mul", ["T8", "ncol"])
    n("t_r", "mul", ["T8", "row"])
    n("t_nr", "mul", ["T8", "nrow"])
    n("TM", "concat", ["t_c", "t_nc", "t_r", "t_nr"])
    n("tb", "gmax", ["TM"])
    n("Z", "conv", ["coords"], w="WRC")
    n("Za", "abs", ["Z"])
    n("Zl", "conv", ["Za"], w="WD", b="bD")
    n("Zm", "min2", ["Zl", "one"])
    n("RC", "relu", ["Zm"])
    n("TRC", "mul", ["T8", "RC"])
    n("tf", "gmax", ["TRC"])
    n("onev", "gmax", ["one"])
    n("newv", "vconcat", ["tf", "tb", "onev"])
    n("mlo", "vslice", ["mem"], start=0, stop=133)
    n("mflag", "vslice", ["mem"], start=132, stop=133)
    n("f2", "vslice", ["mem"], start=133, stop=134)
    n("nf", "dense", ["mflag"], w="Wnf", b="bnf")
    n("gate1", "vmul", ["nf", "has8"])
    n("gate", "vmul", ["gate1", "hg"])
    n("upd", "vmul", ["newv", "gate"])
    n("mlon", "vadd", ["mlo", "upd"])
    n("a6", "vslice", ["aprev"], start=6, stop=7)
    n("pp", "vmul", ["a6", "mflag"])
    n("qq", "vmul", ["pp", "f2"])
    n("f2a", "vadd", ["f2", "pp"])
    n("nq", "dense", ["qq"], w="Wneg")
    n("f2n", "vadd", ["f2a", "nq"])
    n("mem_next", "vconcat", ["mlon", "f2n"])
    n("tfm", "vslice", ["mem_next"], start=0, stop=128)
    n("tbm", "vslice", ["mem_next"], start=128, stop=132)
    n("maps", "mul", ["RC", "tfm"])
    n("rcmap", "conv", ["maps"], w="WB")
    n("rmap", "slice", ["rcmap"], start=0, stop=1)
    n("cmap", "slice", ["rcmap"], start=1, stop=2)
    n("Tmask", "min2", ["rmap", "cmap"])
    n("e8T", "mul", ["e8", "Tmask"])
    n("e8Tg", "mul", ["e8T", "ng"])
    n("e8S", "sub", ["e8", "e8Tg"])
    n("S", "add", ["e8S", "g3"])
    n("s_c", "mul", ["S", "col"])
    n("s_nc", "mul", ["S", "ncol"])
    n("s_r", "mul", ["S", "row"])
    n("s_nr", "mul", ["S", "nrow"])
    n("SM", "concat", ["s_c", "s_nc", "s_r", "s_nr"])
    n("sb", "gmax", ["SM"])
    n("v", "vconcat", ["sb", "tbm"])
    n("pre", "dense", ["v"], w="W1", b="b1")
    n("mvl", "vslice", ["pre"], start=0, stop=4)
    n("mv", "vrelu", ["mvl"])
    n("va", "vslice", ["pre"], start=4, stop=5)
    n("vb", "vslice", ["pre"], start=5, stop=6)
    n("ab", "vmul", ["va", "vb"])
    n("misl", "dense", ["ab"], w="Wm")
    n("mis", "vrelu", ["misl"])
    n("a5", "vslice", ["aprev"], start=5, stop=6)
    n("na5", "dense", ["a5"], w="Wnf", b="bnf")
    n("mis2", "vmul", ["mis", "na5"])
    n("hbg", "vmul", ["hasblk", "grey"])
    n("gf2", "vmul", ["grey", "f2n"])
    n("cv", "vconcat", ["hasblk", "hbg", "gf2"])
    n("nc", "dense", ["cv"], w="Wnc", b="bnc")
    n("u", "vconcat", ["mv", "mis2", "nc"])
    n("logits", "dense", ["u"], w="Wout", b="bout")
    n("ck1", "mul", ["e8", "ng"])
    n("ck2", "mul", ["x4m", "grey"])
    n("click", "add", ["ck1", "ck2"])

    params = {k: np.asarray(v, np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 134, "params": params}

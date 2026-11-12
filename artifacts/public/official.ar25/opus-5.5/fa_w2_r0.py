import numpy as np

def build():
    P = {}
    W1 = np.zeros((5, 16, 3, 3)); b1 = np.zeros(5)
    W1[0, 11, 1, 1] = 1
    W1[1, 4, 1, 0] = 1
    W1[2, 5] = 1; W1[2, 5, 1, 1] = 0; W1[2, 0, 1, 1] = 10; b1[2] = -10
    W1[3, 10] = 1; W1[3, 10, 1, 1] = 0; W1[3, 0, 1, 1] = 10; b1[3] = -10
    W1[4, 10, 1, 1] = 1
    P["W1"] = W1; P["b1"] = b1
    Wr = np.zeros((42, 3, 1, 1)); br = np.zeros(42)
    for k in range(21):
        Wr[k, 0, 0, 0] = 63; br[k] = -(3 * k + 1)
        Wr[21 + k, 1, 0, 0] = 63; br[21 + k] = -(3 * k + 1)
    P["Wr"] = Wr; P["br"] = br
    P["Wt"] = (-2 * np.eye(42)).reshape(42, 42, 1, 1); P["bt"] = np.ones(42)
    Ws = np.zeros((2, 42, 1, 1)); Ws[0, :21] = 1; Ws[1, 21:] = 1
    P["Ws"] = Ws
    NF = 14
    Wb = np.zeros((5, NF)); bb = np.zeros(5)
    Wb[0, 6] = 10; bb[0] = -40
    Wb[1, 7] = 0.5; bb[1] = -30
    Wb[2, 8] = 0.5; bb[2] = -30
    Wb[3, 9] = 1; bb[3] = -25
    Wb[4, 10] = 1; bb[4] = -25
    P["Wb"] = Wb; P["bb"] = bb
    def M(r, c, e):
        m = np.zeros((r, c))
        for (i, j), v in e.items():
            m[i, j] = v
        return m
    P["Wq"] = M(2, 6, {(0, 3): -1, (0, 4): 1, (0, 5): -1, (1, 3): 1, (1, 4): -1, (1, 5): -1})
    P["Wp"] = M(2, 6, {(0, 0): 1, (1, 0): 1})
    P["Wm5"] = M(1, 6, {(0, 0): 1.2 / 63})
    P["Wg"] = M(3, 6, {(0, 1): 1, (1, 2): 1, (2, 1): 1.5 / 63, (2, 2): 1.5 / 63})
    S = 4096.0; Sz = 512.0
    P["Wa1"] = M(3, NF, {(0, 4): S, (1, 2): S, (2, 0): S})
    P["Wa2"] = M(3, NF, {(0, 1): S, (1, 1): S, (2, 1): S})
    P["Wb1"] = M(2, NF, {(0, 5): S, (1, 3): S})
    P["Wb2"] = M(2, NF, {(0, 0): S, (1, 0): S})
    P["Wz2"] = M(3, NF, {(0, 11): Sz, (1, 11): Sz, (2, 11): Sz})
    P["Wz3"] = M(2, NF, {(0, 13): Sz, (1, 12): Sz})
    P["Wd"] = M(3, 5, {(0, 0): 1, (0, 3): -1, (1, 1): 1, (1, 4): -1, (2, 2): 1})
    Wl = np.zeros((8, 6)); bl = np.zeros(8)
    Wl[4, 0] = 1; Wl[4, 3] = 1; Wl[3, 0] = -1; Wl[3, 3] = -1
    Wl[2, 1] = 1; Wl[2, 4] = 1; Wl[1, 1] = -1; Wl[1, 4] = -1
    Wl[5, 2] = 1; Wl[5, 5] = 1
    bl[0] = -1; bl[5] = 1e-4; bl[6] = -1; bl[7] = -1
    P["Wl"] = Wl; P["bl"] = bl

    def n(i, op, ins, **kw):
        d = {"id": i, "op": op, "in": ins}; d.update(kw); return d
    def sl(i, src, a, b, op="slice"):
        return n(i, op, [src], start=a, stop=b)
    g = [
        n("c1", "conv", ["x0"], w="W1", b="b1"), n("K", "relu", ["c1"]),
        n("rc", "conv", ["coords"], w="Wr", b="br"), n("ab", "abs", ["rc"]),
        n("tc", "conv", ["ab"], w="Wt", b="bt"), n("tr", "relu", ["tc"]),
        n("ms", "conv", ["tr"], w="Ws"),
        sl("mr", "ms", 0, 1), sl("mcl", "ms", 1, 2), n("Mc", "mul", ["mr", "mcl"]),
        sl("TR", "K", 0, 2), n("TRm", "mul", ["TR", "Mc"]),
        sl("rowc", "coords", 0, 1), sl("colc", "coords", 1, 2),
        n("TRr", "mul", ["TRm", "rowc"]), n("TRc", "mul", ["TRm", "colc"]),
        n("g1", "gmean", ["TRm"]), n("g2", "gmean", ["TRr"]), n("g3", "gmean", ["TRc"]),
        sl("Z5", "K", 2, 3), n("gz5", "gmax", ["Z5"]),
        sl("ZA", "K", 3, 4), n("cza", "colsum", ["ZA"]), n("gcza", "gmax", ["cza"]),
        n("rza", "rowsum", ["ZA"]), n("grza", "gmax", ["rza"]),
        sl("AA", "K", 4, 5), n("ca", "colsum", ["AA"]), n("gca", "gmax", ["ca"]),
        n("ra", "rowsum", ["AA"]), n("gra", "gmax", ["ra"]),
        n("ZAr", "mul", ["ZA", "rowc"]), n("ZAc", "mul", ["ZA", "colc"]),
        n("gzn", "gmean", ["ZA"]), n("gzr", "gmean", ["ZAr"]), n("gzc", "gmean", ["ZAc"]),
        n("f", "vconcat", ["g1", "g2", "g3", "gz5", "gcza", "grza", "gca", "gra", "gzn", "gzr", "gzc"]),
        n("bpre", "dense", ["f"], w="Wb", b="bb"), n("bl5", "vsigmoid", ["bpre"]),
        sl("Vv", "bl5", 3, 4, "vslice"), sl("Hv", "bl5", 4, 5, "vslice"),
        n("VH", "vmul", ["Vv", "Hv"]), n("bv", "vconcat", ["bl5", "VH"]),
        n("q", "dense", ["bv"], w="Wq"), n("p2", "dense", ["bv"], w="Wp"),
        n("sxy", "vmul", ["p2", "q"]), n("m5", "dense", ["bv"], w="Wm5"),
        n("gm", "dense", ["bv"], w="Wg"),
        n("a1", "dense", ["f"], w="Wa1"), n("a2", "dense", ["f"], w="Wa2"),
        n("b1v", "dense", ["f"], w="Wb1"), n("b2v", "dense", ["f"], w="Wb2"),
        n("p1", "vmul", ["a1", "a2"]), n("p2b", "vmul", ["b1v", "b2v"]),
        n("pp", "vconcat", ["p1", "p2b"]), n("D", "dense", ["pp"], w="Wd"),
        sl("Dxy", "D", 0, 2, "vslice"), sl("Nv", "D", 2, 3, "vslice"),
        n("A", "vmul", ["Dxy", "sxy"]), n("thr5", "vmul", ["m5", "Nv"]),
        n("z2", "dense", ["f"], w="Wz2"), n("z3", "dense", ["f"], w="Wz3"),
        n("pz1", "vmul", ["a1", "z2"]), n("pz2", "vmul", ["z3", "b2v"]),
        n("ppz", "vconcat", ["pz1", "pz2"]), n("E", "dense", ["ppz"], w="Wd"),
        n("Em", "vmul", ["E", "gm"]),
        n("lin", "vconcat", ["A", "thr5", "Em"]),
        n("logits", "dense", ["lin"], w="Wl", b="bl"),
        sl("click", "TRm", 0, 1),
    ]
    params = {k: np.asarray(v, np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 0, "params": params}

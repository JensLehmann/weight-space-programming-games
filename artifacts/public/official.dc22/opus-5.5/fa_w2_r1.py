import numpy as np

def build():
    P = {}
    K = np.zeros((5, 16, 7, 7)); kb = np.zeros(5)
    for c in [1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15]:
        K[0, c, 3, 3] = 1
    for c in [0, 3, 4, 5]:
        K[1, c, 3, 3] = -1
    kb[1] = 1
    K[2, 11, 3, 3] = 1
    K[3, 5, 2:5, 2:5] = 1
    K[4, 5, :, :] = 1
    P["K"] = K; P["kb"] = kb
    P["Dil"] = np.ones((1, 1, 7, 7))
    Ws = np.zeros((8, 2, 3, 3))
    for k, (i, j) in enumerate([(0, 1), (2, 1), (1, 0), (1, 2)]):
        for ch in range(2):
            Ws[2 * k + ch, ch, i, j] = 0.98
    P["Ws"] = Ws
    P["Wv"] = np.array([1e4, 1.0]).reshape(1, 2, 1, 1)
    Wd = np.zeros((4, 1, 5, 5))
    for k, (i, j) in enumerate([(0, 2), (4, 2), (2, 0), (2, 4)]):
        Wd[k, 0, i, j] = 1
    P["Wd"] = Wd
    Wo = np.zeros((8, 6)); bo = np.zeros(8)
    for k in range(4):
        Wo[1 + k, k] = 1
    Wo[6, 4] = 1; Wo[6, 5] = 1e6; bo[6] = 1e-4
    bo[0] = bo[5] = bo[7] = -1e6
    P["Wo"] = Wo; P["bo"] = bo
    P["Wmem"] = np.array([[1.0, 1.0, -2.0]])
    P["Wq"] = np.array([[-2.0]]); P["bq"] = np.array([1.0])
    P["Wcol"] = np.array([0.0, 0.3, 2.0]).reshape(1, 3, 1, 1)
    P["Wrow"] = np.array([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1)
    P["Wdf"] = np.array([[1, 0, -1, 0], [-1, 0, 1, 0], [0, 1, 0, -1], [0, -1, 0, 1]], dtype=np.float64)
    Wmv = np.zeros((1, 8)); Wmv[0, 1:5] = 1
    P["Wmv"] = Wmv
    P["Wst"] = np.array([[1.0, -63.0, -63.0, -63.0, -63.0]])
    def n(i, op, inp, **kw):
        d = {"id": i, "op": op, "in": inp}; d.update(kw); return d
    def sl(i, src, a, b):
        return n(i, "slice", [src], start=a, stop=b)
    g = [
        n("m4", "conv", ["x0"], w="K", b="kb"),
        sl("Wm", "m4", 0, 1), sl("nonbg", "m4", 1, 2), sl("c11", "m4", 2, 3),
        sl("n5", "m4", 3, 4), sl("n7", "m4", 4, 5), sl("one", "coords", 2, 3),
        n("gs", "sub", ["c11", "n5"]), n("G", "relu", ["gs"]),
        n("n5b", "min2", ["n7", "one"]), n("Bm", "mul", ["nonbg", "n5b"]),
        n("dl", "conv", ["Wm"], w="Dil"), n("R", "min2", ["dl", "one"]),
        n("M2", "concat", ["Wm", "R"]), n("G2", "concat", ["G", "G"]),
        {"id": "H", "op": "loop", "T": 200,
         "state": [{"id": "h", "zeros": [2, 64, 64]}],
         "body": [
             n("sh", "conv", ["h"], w="Ws"),
             sl("s0", "sh", 0, 2), sl("s1", "sh", 2, 4), sl("s2", "sh", 4, 6), sl("s3", "sh", 6, 8),
             n("x01", "max2", ["s0", "s1"]), n("x23", "max2", ["s2", "s3"]),
             n("xa", "max2", ["x01", "x23"]), n("xg", "max2", ["xa", "G2"]),
             n("hn", "mul", ["xg", "M2"])],
         "next": {"h": "hn"}},
        sl("h1", "H.h", 0, 1), sl("h2", "H.h", 1, 2),
        n("h2w", "mul", ["h2", "Wm"]), n("hc", "concat", ["h1", "h2w"]),
        n("V", "conv", ["hc"], w="Wv"), sl("Pl", "x0", 14, 15),
        n("D", "conv", ["V"], w="Wd"), n("PD", "mul", ["D", "Pl"]), n("dirs", "gmax", ["PD"]),
        n("PV", "mul", ["V", "Pl"]), n("cur", "gmax", ["PV"]),
        sl("rc0", "coords", 0, 1), sl("cc0", "coords", 1, 2),
        n("pr", "mul", ["Pl", "rc0"]), n("pc", "mul", ["Pl", "cc0"]),
        n("pp", "concat", ["pr", "pc"]), n("ppos", "gmax", ["pp"]),
        n("prev", "vslice", ["mem"], start=1, stop=3),
        n("pq", "vconcat", ["ppos", "prev"]), n("dfl", "dense", ["pq"], w="Wdf"),
        n("df", "vrelu", ["dfl"]), n("mvd", "dense", ["aprev"], w="Wmv"),
        n("sv", "vconcat", ["mvd", "df"]), n("stl", "dense", ["sv"], w="Wst"),
        n("stuck", "vrelu", ["stl"]),
        n("lv", "vconcat", ["dirs", "cur", "stuck"]),
        n("logits", "dense", ["lv"], w="Wo", b="bo"),
        n("m0", "vslice", ["mem"], start=0, stop=1),
        n("a6", "vslice", ["aprev"], start=6, stop=7),
        n("ma", "vmul", ["m0", "a6"]), n("mv", "vconcat", ["m0", "a6", "ma"]),
        n("par", "dense", ["mv"], w="Wmem"),
        n("mem_next", "vconcat", ["par", "ppos"]),
        n("q", "dense", ["par"], w="Wq", b="bq"),
        n("rowc", "conv", ["coords"], w="Wrow"), n("colc", "conv", ["coords"], w="Wcol"),
        n("qr", "mul", ["rowc", "q"]), n("sc", "add", ["qr", "colc"]),
        n("click", "mul", ["sc", "Bm"]),
    ]
    params = {k: np.asarray(v, np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 3, "params": params}

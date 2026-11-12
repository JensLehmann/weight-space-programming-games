import numpy as np

def build():
    P = {}
    f = lambda a: np.array(a, np.float32)
    P["Wra"] = f([126, 0, -19]).reshape(1, 3, 1, 1)
    P["Wrb"] = f([-126, 0, 125]).reshape(1, 3, 1, 1)
    cols = [10, 6, 15, 11, 12, 8]; areas = [1, 4, 9, 16, 25, 49]
    W6 = np.zeros((6, 16, 1, 1), np.float32)
    for k, c in enumerate(cols): W6[k, c, 0, 0] = 1.0
    P["W6"] = W6
    Ws = np.zeros((6, 6), np.float32); Wh = np.zeros((6, 6), np.float32)
    for k in range(6):
        Ws[k, k] = 1; Ws[k, :k] = -1
        Wh[k, k] = 1; Wh[k, k + 1:] = -1
    P["Ws"] = Ws; P["Wh"] = Wh
    P["Wcnt"] = np.diag([4096.0 / a for a in areas]).astype(np.float32)
    P["bcnt"] = -np.ones(6, np.float32)
    P["Wneg"] = -np.eye(6, dtype=np.float32); P["b1"] = np.ones(6, np.float32)
    P["Wone"] = np.ones((1, 6), np.float32)
    P["Wg0"] = f([[-1.0]]); P["bg0"] = f([1.0])
    P["Wsum"] = np.ones((1, 6, 1, 1), np.float32)
    wt = np.zeros(16, np.float32); wt[9] = 1.0; wt[2] = 1.0
    P["Wt"] = wt.reshape(1, 16, 1, 1)
    P["K3"] = np.ones((1, 1, 3, 3), np.float32)
    P["K7"] = np.ones((1, 1, 7, 7), np.float32)
    P["K5"] = np.ones((1, 1, 5, 5), np.float32)
    P["Wna"] = f([-1.0]).reshape(1, 1, 1, 1); P["bna"] = f([100.0])
    P["Wtb"] = f([0.6, 0.006, 0.0]).reshape(1, 3, 1, 1)
    P["Wq"] = f([20000.0]).reshape(1, 1, 1, 1); P["bq"] = f([1.0])
    P["Wcl"] = f([1.0, 200.0, 0.001]).reshape(1, 3, 1, 1)
    P["Wlg"] = np.zeros((8, 8), np.float32)
    bl = np.zeros(8, np.float32); bl[6] = 1.0; P["blg"] = bl

    def dl(lid, init, acc):
        return {"id": lid, "op": "loop", "T": 64,
                "state": [{"id": lid + "d", "init": init}, {"id": acc, "zeros": [1, 64, 64]}],
                "body": [{"id": lid + "dc", "op": "conv", "in": [lid + "d"], "w": "K3"},
                         {"id": lid + "dn", "op": "min2", "in": [lid + "dc", "one"]},
                         {"id": lid + "an", "op": "add", "in": [acc, lid + "dn"]}],
                "next": {lid + "d": lid + "dn", acc: lid + "an"}}

    def n(i, op, inp, **k):
        d = {"id": i, "op": op, "in": inp}; d.update(k); return d

    g = [
        n("one", "slice", ["coords"], start=2, stop=3),
        n("ra", "conv", ["coords"], w="Wra"), n("ra1", "relu", ["ra"]), n("ra2", "min2", ["ra1", "one"]),
        n("rb", "conv", ["coords"], w="Wrb"), n("rb1", "relu", ["rb"]), n("rb2", "min2", ["rb1", "one"]),
        n("rm", "mul", ["ra2", "rb2"]),
        n("C6r", "conv", ["x0"], w="W6"), n("C6", "mul", ["C6r", "rm"]),
        n("p", "gmax", ["C6"]), n("pm", "gmean", ["C6"]),
        n("ec", "dense", ["pm"], w="Wcnt", b="bcnt"), n("e1", "vrelu", ["ec"]),
        n("e2l", "dense", ["e1"], w="Wneg", b="b1"), n("e2", "vrelu", ["e2l"]),
        n("el", "dense", ["e2"], w="Wneg", b="b1"),
        n("sl", "dense", ["el"], w="Ws"), n("s", "vrelu", ["sl"]),
        n("gsum", "dense", ["s"], w="Wone"),
        n("g0", "dense", ["gsum"], w="Wg0", b="bg0"),
        n("hl", "dense", ["p"], w="Wh"), n("h", "vrelu", ["hl"]),
        n("hm", "vmul", ["h", "g0"]), n("sel", "vadd", ["s", "hm"]),
        n("Ms", "mul", ["C6", "sel"]), n("M", "conv", ["Ms"], w="Wsum"),
        n("Tr", "conv", ["x0"], w="Wt"), n("T", "mul", ["Tr", "rm"]),
        dl("L", "T", "acc"),
        n("na", "conv", ["L.acc"], w="Wna", b="bna"),
        n("tb", "conv", ["coords"], w="Wtb"), n("nb", "add", ["na", "tb"]),
        n("q", "mul", ["M", "nb"]), n("gq", "gmax", ["q"]), n("qs", "sub", ["q", "gq"]),
        n("qb", "conv", ["qs"], w="Wq", b="bq"), n("qr", "relu", ["qb"]),
        n("Ffar", "min2", ["qr", "one"]),
        {"id": "G", "op": "loop", "T": 8, "state": [{"id": "fr", "init": "Ffar"}],
         "body": [n("frc", "conv", ["fr"], w="K3"), n("frm", "min2", ["frc", "one"]),
                  n("frn", "mul", ["frm", "M"])], "next": {"fr": "frn"}},
        n("oth0", "sub", ["M", "G.fr"]), n("oth", "relu", ["oth0"]),
        dl("L2", "oth", "acc2"),
        n("d1c", "conv", ["G.fr"], w="K7"), n("d1", "min2", ["d1c", "one"]),
        n("d2c", "conv", ["d1"], w="K5"), n("d2", "min2", ["d2c", "one"]),
        n("D2", "mul", ["d2", "rm"]),
        n("f1", "mul", ["L2.acc2", "gsum"]), n("f0", "mul", ["L.acc", "g0"]),
        n("fld", "add", ["f1", "f0"]),
        n("cc", "concat", ["fld", "D2", "L.acc"]),
        n("click", "conv", ["cc"], w="Wcl"),
        n("logits", "dense", ["avail"], w="Wlg", b="blg"),
    ]
    return {"graph": g, "memory_size": 0, "params": P}

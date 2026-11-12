import numpy as np

def build():
    P = {}
    def c1(name, w, b=None):
        w = np.asarray(w, np.float32)
        P[name] = w.reshape(w.shape[0], w.shape[1], 1, 1)
        if b is not None:
            P["b" + name] = np.asarray(b, np.float32)
    def vd(name, w, b=None):
        P[name] = np.asarray(w, np.float32)
        if b is not None:
            P["b" + name] = np.asarray(b, np.float32)

    w = np.zeros((4, 19), np.float32)
    w[0, 8] = 1.0; w[1, 14] = 1.0
    w[2, 17] = 126.0; w[3, 16] = 126.0
    c1("Wbv", w, [0.0, 0.0, -1.0, -3.0])
    c1("Wbm", [[1, 0, 1, 1], [0, 1, 1, 1]], [-2.0, -2.0])
    wan = np.zeros((2, 2, 5, 5), np.float32)
    for c in range(2):
        wan[c, c, 2, 2] = 1.0
        for (i, j) in [(1, 1), (1, 2), (2, 1), (1, 3), (1, 4)]:
            wan[c, c, i, j] = -1.0
    P["Wan"] = wan
    c1("WA", [[1.0, 1.0]])
    c1("WI", [[4032.0, 63.0, 0.0]])

    # tile detection by block erosion/dilation
    we2 = np.zeros((2, 2, 3, 3), np.float32)
    wd2 = np.zeros((2, 2, 3, 3), np.float32)
    we4 = np.zeros((2, 2, 7, 7), np.float32)
    wd4 = np.zeros((2, 2, 7, 7), np.float32)
    for c in range(2):
        we2[c, c, 1:3, 1:3] = 1.0
        wd2[c, c, 0:2, 0:2] = 1.0
        we4[c, c, 3:7, 3:7] = 1.0
        wd4[c, c, 0:4, 0:4] = 1.0
    P["Wer2"] = we2; P["bWer2"] = np.full((2,), -3.0, np.float32)
    P["Wdl2"] = wd2
    P["Wer4"] = we4; P["bWer4"] = np.full((2,), -15.0, np.float32)
    P["Wdl4"] = wd4
    c1("Ws2", [[1.0, 1.0]])

    wsh = np.zeros((8, 2, 3, 3), np.float32)
    for c in range(2):
        for k, (i, j) in enumerate([(0, 1), (2, 1), (1, 0), (1, 2)]):
            wsh[2 * k + c, c, i, j] = 1.0
    P["Wsh"] = wsh
    P["bWsh"] = np.full((8,), -1.0, np.float32)
    c1("W0", [[160.0, 0.0], [0.0, 160.0]])
    vd("WS", [[4096.0, 4096.0]])
    vd("WG", [[4096.0, 12288.0]])

    K = 8
    vd("Wd", [[0, -1, 0, 0, 1, 0], [0, 1, 0, 0, -1, 0],
              [0, 0, -1, 0, 0, 1], [0, 0, 1, 0, 0, -1]])
    vd("Wc2", np.eye(4), -np.ones(4))
    vd("Wcl", np.concatenate([np.eye(4), -np.eye(4)], 1))
    vd("We", [[-1, -1, 0, 0]], [1.0])
    vd("Wmv", [[0, 0, 1, 1]])
    vd("Wup", [[1, 0, 0, 0]])
    vd("Wg1", [[1.0]], [-(K - 1.0)])
    vd("Wg2", [[1.0]], [-1.0])
    vd("Wnov", [[-1.0, 1.0]], [1.0])
    vd("Wc1", [[1.0]], [1.0])

    c1("Ws", [[-2.0]], [1.0])
    c1("Wkn", [[1.0, 8192.0]])
    c1("Wng", [[-1.0]], [20000.0])
    c1("Wsl", [[2.0]], [1.0])
    P["Wlog"] = np.zeros((8, 8), np.float32)
    bl = np.zeros(8, np.float32); bl[6] = 1.0
    P["blog"] = bl

    def cv(i, x, wn_, b=True):
        d = {"id": i, "op": "conv", "in": [x], "w": wn_}
        if b:
            d["b"] = "b" + wn_
        return d
    def op(i, o, ins, **kw):
        d = {"id": i, "op": o, "in": ins}
        d.update(kw)
        return d
    def dn(i, x, wn_, b=True):
        d = {"id": i, "op": "dense", "in": [x], "w": wn_}
        if b:
            d["b"] = "b" + wn_
        return d

    body = [
        cv("sh", "w", "Wsh"),
        op("q1", "slice", ["sh"], start=0, stop=2),
        op("q2", "slice", ["sh"], start=2, stop=4),
        op("q3", "slice", ["sh"], start=4, stop=6),
        op("q4", "slice", ["sh"], start=6, stop=8),
        op("a1", "max2", ["q1", "q2"]),
        op("a2", "max2", ["q3", "q4"]),
        op("a3", "max2", ["a1", "a2"]),
        op("wnx", "max2", ["a3", "w"]),
    ]

    g = [
        op("xc", "concat", ["x0", "coords"]),
        cv("bv", "xc", "Wbv"),
        op("bvr", "relu", ["bv"]),
        op("one", "slice", ["coords"], start=2, stop=3),
        op("bvc", "min2", ["bvr", "one"]),
        cv("bm", "bvc", "Wbm"),
        op("bmr", "relu", ["bm"]),
        {"id": "an", "op": "conv", "in": ["bmr"], "w": "Wan"},
        op("A2", "relu", ["an"]),
        cv("A", "A2", "WA", b=False),
        cv("Ii", "coords", "WI", b=False),
        op("m", "slice", ["x0"], start=11, stop=13),
        cv("er2", "m", "Wer2"),
        op("er2r", "relu", ["er2"]),
        cv("dl2", "er2r", "Wdl2", b=False),
        op("t1", "min2", ["dl2", "one"]),
        cv("er4", "m", "Wer4"),
        op("er4r", "relu", ["er4"]),
        cv("dl4", "er4r", "Wdl4", b=False),
        op("t2", "min2", ["dl4", "one"]),
        cv("e4s", "er4r", "Ws2", b=False),
        op("s2", "gmax", ["e4s"]),
        op("dtt", "sub", ["t2", "t1"]),
        op("es", "mul", ["dtt", "s2"]),
        op("tile", "add", ["t1", "es"]),
        op("corner", "sub", ["m", "tile"]),
        cv("Wz", "tile", "W0", b=False),
        {"id": "DT", "op": "loop", "T": 128,
         "state": [{"id": "w", "init": "Wz"}],
         "body": body, "next": {"w": "wnx"}},
        op("cw", "mul", ["corner", "DT.w"]),
        op("gs", "gmean", ["cw"]),
        dn("S", "gs", "WS", b=False),
        op("tI", "mul", ["tile", "Ii"]),
        op("gG", "gmean", ["tI"]),
        dn("G", "gG", "WG", b=False),
        op("me", "vconcat", ["mem", "S", "G"]),
        dn("dv", "me", "Wd", b=False),
        op("r", "vrelu", ["dv"]),
        dn("r2", "r", "Wc2"),
        op("r3", "vrelu", ["r2"]),
        op("rr", "vconcat", ["r", "r3"]),
        dn("cf", "rr", "Wcl", b=False),
        dn("eq", "cf", "We"),
        dn("mvf", "cf", "Wmv", b=False),
        dn("upv", "cf", "Wup", b=False),
        op("t", "vmul", ["eq", "mvf"]),
        op("cvv", "vslice", ["mem"], start=3, stop=4),
        dn("g1l", "cvv", "Wg1"),
        op("g1", "vrelu", ["g1l"]),
        dn("g2l", "g1", "Wg2"),
        op("g2", "vrelu", ["g2l"]),
        op("g12", "vconcat", ["g1", "g2"]),
        dn("nov", "g12", "Wnov"),
        op("tc", "vmul", ["t", "nov"]),
        op("imp", "vadd", ["upv", "tc"]),
        dn("cp1", "cvv", "Wc1"),
        op("cn", "vmul", ["tc", "cp1"]),
        op("Lv", "vslice", ["mem"], start=0, stop=1),
        op("dI", "sub", ["Ii", "Lv"]),
        op("ad", "abs", ["dI"]),
        cv("sp", "dI", "Ws"),
        op("spr", "relu", ["sp"]),
        op("sc", "min2", ["spr", "one"]),
        op("knin", "concat", ["dI", "sc"]),
        cv("kn", "knin", "Wkn", b=False),
        op("df", "sub", ["ad", "kn"]),
        op("tt", "mul", ["df", "imp"]),
        op("key", "add", ["kn", "tt"]),
        cv("ng", "key", "Wng"),
        op("click", "mul", ["A", "ng"]),
        op("gm", "gmax", ["click"]),
        op("dd", "sub", ["click", "gm"]),
        cv("sl", "dd", "Wsl"),
        op("sel", "relu", ["sl"]),
        op("sI", "mul", ["sel", "Ii"]),
        op("Lnew", "gmax", ["sI"]),
        op("mem_next", "vconcat", ["Lnew", "S", "G", "cn"]),
        {"id": "logits", "op": "dense", "in": ["avail"], "w": "Wlog", "b": "blog"},
    ]
    return {"graph": g, "memory_size": 4, "params": P}

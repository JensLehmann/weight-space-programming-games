import numpy as np

def build():
    G = []
    P = {}

    def nd(i, op, ins, **kw):
        d = {"id": i, "op": op, "in": ins}
        d.update(kw)
        return d

    def add(i, op, ins, **kw):
        G.append(nd(i, op, ins, **kw))
        return i

    def sk(dy, dx, c):
        w = np.zeros((c, c, 3, 3), np.float32)
        for k in range(c):
            w[k, k, 1 + dy, 1 + dx] = 1.0
        return w

    P["SL"] = sk(0, 1, 1); P["SR"] = sk(0, -1, 1)
    P["SU"] = sk(1, 0, 1); P["SD"] = sk(-1, 0, 1)
    P["SL3"] = sk(0, 1, 3); P["SR3"] = sk(0, -1, 3)
    P["SU3"] = sk(1, 0, 3); P["SD3"] = sk(-1, 0, 3)
    adj = np.zeros((2, 2, 3, 3), np.float32)
    adj[0, 0, 1, 1] = 1; adj[0, 1, 0, 1] = 1; adj[0, 1, 2, 1] = 1
    adj[1, 0, 1, 1] = 1; adj[1, 1, 1, 0] = 1; adj[1, 1, 1, 2] = 1
    P["ADJ"] = adj
    P["ADJb"] = np.array([-1.5, -1.5], np.float32)
    P["K10"] = np.array([[10.0]], np.float32)
    P["K20"] = (20.0 * np.eye(2)).astype(np.float32)
    P["NEG1"] = np.array([[-1.0]], np.float32)
    P["ONE1"] = np.array([1.0], np.float32)

    add("one", "slice", ["coords"], start=2, stop=3)
    for c in (4, 6, 8, 9, 11):
        add("x%d" % c, "slice", ["x0"], start=c, stop=c + 1)
    add("x46", "concat", ["x4", "x6"])
    add("adjc", "conv", ["x46"], w="ADJ", b="ADJb")
    add("adjr", "relu", ["adjc"])
    add("adjg", "gmax", ["adjr"])
    add("adjd", "dense", ["adjg"], w="K20")
    add("gates", "vtanh", ["adjd"])
    add("gH", "vslice", ["gates"], start=0, stop=1)
    add("gV", "vslice", ["gates"], start=1, stop=2)

    def axis(t, pmax, psum, Sneg, Spos, Sneg3, Spos3):
        C = add(t + "C", psum, ["x11"])
        Pp = add(t + "P", pmax, ["x9"])
        D = add(t + "D", pmax, ["x6"])
        O = add(t + "O", pmax, ["x8"])
        for nm, S in (("Lw", Spos), ("Rw", Sneg)):
            G.append({"id": t + nm, "op": "loop", "T": 6,
                      "state": [{"id": t + nm + "s", "init": C},
                                {"id": t + nm + "m", "zeros": [1, 64, 64]}],
                      "body": [nd(t + nm + "s2", "conv", [t + nm + "s"], w=S),
                               nd(t + nm + "m2", "max2", [t + nm + "m", t + nm + "s2"])],
                      "next": {t + nm + "s": t + nm + "s2", t + nm + "m": t + nm + "m2"}})
        add(t + "ar", "relu", [C])
        add(t + "a", "min2", [t + "ar", "one"])
        for nm in ("Lw", "Rw"):
            add(t + nm + "d", "sub", [t + nm + "." + t + nm + "m", C])
            add(t + nm + "dr", "relu", [t + nm + "d"])
            add(t + nm + "i", "min2", [t + nm + "dr", "one"])
        add(t + "h1", "mul", [t + "a", t + "Lwi"])
        H = add(t + "H", "mul", [t + "h1", t + "Rwi"])
        add(t + "nH", "sub", ["one", H])
        add(t + "DnH", "mul", [D, t + "nH"])
        add(t + "nO", "sub", ["one", O])
        U = add(t + "U", "mul", [t + "DnH", t + "nO"])
        add(t + "Hn", "conv", [H], w=Sneg)
        add(t + "Hp", "conv", [H], w=Spos)
        add(t + "nHn", "sub", ["one", t + "Hn"])
        add(t + "nHp", "sub", ["one", t + "Hp"])
        Hr = add(t + "Hr", "mul", [H, t + "nHn"])
        Hl = add(t + "Hl", "mul", [H, t + "nHp"])
        add(t + "nP", "sub", ["one", Pp])
        add(t + "Pn", "conv", [Pp], w=Sneg)
        add(t + "Pq", "conv", [Pp], w=Spos)
        E = add(t + "E", "mul", [t + "Pn", t + "nP"])
        F = add(t + "F", "mul", [t + "Pq", t + "nP"])
        X = add(t + "X", "concat", [E, F, Pp])
        GG = add(t + "G", "concat", [Hr, Hl, D])

        def mn(p, src):
            return [nd(p + "pr", "mul", [src, GG]),
                    nd(p + "g", "gmax", [p + "pr"]),
                    nd(p + "g0", "vslice", [p + "g"], start=0, stop=1),
                    nd(p + "g1", "vslice", [p + "g"], start=1, stop=2),
                    nd(p + "g2", "vslice", [p + "g"], start=2, stop=3),
                    nd(p + "t", "vmul", [p + "g0", p + "g1"]),
                    nd(p + "t2", "vmul", [p + "t", p + "g2"]),
                    nd(p + "ps", "slice", [src], start=2, stop=3),
                    nd(p + "q", "mul", [U, p + "ps"]),
                    nd(p + "ug", "gmax", [p + "q"]),
                    nd(p + "t3", "vmul", [p + "t2", p + "ug"])]

        G.extend(mn(t + "z", X))
        m0 = t + "zt3"
        res = []
        for nm, S3 in (("N", Sneg3), ("Q", Spos3)):
            L = t + "lp" + nm
            body = [nd(L + "xs2", "conv", [L + "xs"], w=S3)] + mn(L + "b", L + "xs2")
            body.append(nd(L + "acc2", "vadd", [L + "acc", L + "bt3"]))
            G.append({"id": L, "op": "loop", "T": 63,
                      "state": [{"id": L + "xs", "init": X},
                                {"id": L + "acc", "zeros": [1]}],
                      "body": body,
                      "next": {L + "xs": L + "xs2", L + "acc": L + "acc2"}})
            add(L + "k", "dense", [L + "." + L + "acc"], w="K10")
            res.append(add(L + "s", "vtanh", [L + "k"]))
        add(t + "rem", "mul", [U, t + "nP"])
        add(t + "rg", "gmax", [t + "rem"])
        add(t + "rf", "dense", [t + "rg"], w="NEG1", b="ONE1")
        dn = add(t + "dn", "vmul", [t + "rf", m0])
        return m0, res[0], res[1], dn

    h0, hN, hQ, hd = axis("h", "colmax", "colsum", "SL", "SR", "SL3", "SR3")
    v0, vN, vQ, vd = axis("v", "rowmax", "rowsum", "SU", "SD", "SU3", "SD3")

    add("fhN", "vmul", ["gH", hN]); add("fhQ", "vmul", ["gH", hQ])
    add("gh0", "vmul", ["gH", h0])
    add("fvN", "vmul", ["gV", vN]); add("fvQ", "vmul", ["gV", vQ])
    add("gv0", "vmul", ["gV", v0])
    add("ghd", "vmul", ["gH", hd]); add("gvd", "vmul", ["gV", vd])
    add("has8", "gmax", ["x8"])

    add("a6", "vslice", ["aprev"], start=6, stop=7)
    add("mem_next", "vadd", ["mem", "a6"])
    P["K20s"] = np.array([[20.0]], np.float32)
    P["Bl"] = np.array([-130.0], np.float32)
    add("lated", "dense", ["mem_next"], w="K20s", b="Bl")
    add("late", "vsigmoid", ["lated"])

    add("feat", "vconcat", ["fhN", "fhQ", "gh0", "fvN", "fvQ", "gv0",
                            "has8", "late", "ghd", "gvd"])
    Wo = np.zeros((8, 10), np.float32)
    for r in (1, 2, 3, 4):
        Wo[r, 2] = -3.0; Wo[r, 5] = -3.0
    Wo[1, 3] = 2.0; Wo[2, 4] = 1.5; Wo[3, 0] = 2.0; Wo[4, 1] = 1.5
    Wo[5, 8] = 3.0; Wo[5, 9] = 3.0; Wo[5, 7] = 2.0
    Wo[6, 6] = 1.0
    for j in (0, 1, 3, 4):
        Wo[6, j] = -1.5
    Wo[6, 2] = 3.0; Wo[6, 5] = 3.0
    Wo[6, 8] = -5.0; Wo[6, 9] = -5.0; Wo[6, 7] = -3.0
    P["Wo"] = Wo
    P["bo"] = np.array([-10, 0, 0, 0, 0, 0.2, 0, -10], np.float32)
    add("logits", "dense", ["feat"], w="Wo", b="bo")

    P["CB"] = np.array([0.5, 0.0, 3.0], np.float32).reshape(1, 3, 1, 1)
    add("cbase", "conv", ["coords"], w="CB")
    add("ca", "mul", ["hD", "gH"])
    add("cb", "mul", ["vD", "gV"])
    add("cs1", "sub", ["cbase", "ca"])
    add("cs2", "sub", ["cs1", "cb"])
    add("click", "mul", ["x8", "cs2"])

    return {"graph": G, "memory_size": 1, "params": P}

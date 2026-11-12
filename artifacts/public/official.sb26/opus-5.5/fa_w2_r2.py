import numpy as np

def build():
    G = []; P = {}
    def N(i, op, ins, **k):
        d = {"id": i, "op": op, "in": ins}; d.update(k); G.append(d); return i
    f = np.float32
    Wcc = np.zeros((4, 3, 1, 1), f); Wcc[:, 0, 0, 0] = [63, 126, -126, 126]
    P["Wcc"] = Wcc; P["bcc"] = np.array([-1, -15, 101, -107], f)
    N("cc", "conv", ["coords"], w="Wcc", b="bcc")
    N("one", "slice", ["coords"], start=2, stop=3)
    N("c0", "slice", ["cc"], start=0, stop=1); N("c0a", "abs", ["c0"])
    P["Wr1"] = np.full((1, 1, 1, 1), -2, f); P["br1"] = np.array([1.5], f)
    N("r1a", "conv", ["c0a"], w="Wr1", b="br1"); N("r1b", "relu", ["r1a"])
    N("row1", "min2", ["r1b", "one"])
    N("c1", "slice", ["cc"], start=1, stop=2); N("c2", "slice", ["cc"], start=2, stop=3)
    N("rgm", "min2", ["c1", "c2"]); N("rgr", "relu", ["rgm"]); N("reg", "min2", ["rgr", "one"])
    N("c3", "slice", ["cc"], start=3, stop=4); N("pr0", "relu", ["c3"]); N("pal", "min2", ["pr0", "one"])
    Wk = np.zeros((2, 16, 7, 7), f)
    for c in range(16):
        if c not in (0, 2, 3, 4, 5): Wk[0, c, 3, 3] = 1
        if c not in (0, 3, 4, 5): Wk[1, c, 3, 3] = 1
    Wk[0, 5, 3, 2] = 1
    Wk[1, 4, 3, 2] = 1; Wk[1, 4, 0, 3] = 1; Wk[1, 4, 6, 3] = 1
    P["Wk"] = Wk; P["bk"] = np.array([-1, -3], f)
    N("mk", "conv", ["x0"], w="Wk", b="bk"); N("mr", "relu", ["mk"])
    N("mt0", "slice", ["mr"], start=0, stop=1); N("mt", "mul", ["mt0", "row1"])
    N("ms0", "slice", ["mr"], start=1, stop=2); N("ms", "mul", ["ms0", "reg"])
    N("cmt", "colmax", ["mt"]); N("cms", "colmax", ["ms"]); N("Mc", "concat", ["cmt", "cms"])
    Wsh = np.zeros((2, 2, 3, 3), f); Wsh[0, 0, 1, 0] = 1; Wsh[1, 1, 1, 0] = 1
    P["Wsh"] = Wsh
    G.append({"id": "PL", "op": "loop", "T": 64,
              "state": [{"id": "p", "zeros": [2, 64, 64]}],
              "body": [{"id": "pc", "op": "conv", "in": ["p"], "w": "Wsh"},
                       {"id": "pn", "op": "add", "in": ["pc", "Mc"]}],
              "next": {"p": "pn"}})
    cv = np.zeros((16, 16, 1, 1), f)
    for c in range(16):
        if c not in (0, 2, 3, 4, 5): cv[c, c, 0, 0] = 1
    P["Wcv"] = cv
    N("R1p", "conv", ["x0"], w="Wcv"); N("R1", "mul", ["R1p", "row1"])
    P["Id2"] = np.eye(2, dtype=f).reshape(2, 2, 1, 1)
    P["NId2"] = (-np.eye(2, dtype=f)).reshape(2, 2, 1, 1)
    P["one2"] = np.ones(2, f)
    P["neg16"] = -np.ones((1, 16), f); P["b1"] = np.ones(1, f)
    NI = 12
    mis = []; tcs = []; msbs = []
    for i in range(NI):
        P[f"bi{i}"] = np.full(2, -(i + 1), f)
        N(f"dt{i}", "conv", ["PL.p"], w="Id2", b=f"bi{i}")
        N(f"da{i}", "abs", [f"dt{i}"])
        N(f"bd{i}", "conv", [f"da{i}"], w="NId2", b="one2")
        N(f"br{i}", "relu", [f"bd{i}"])
        N(f"bt{i}", "slice", [f"br{i}"], start=0, stop=1)
        N(f"bs{i}", "slice", [f"br{i}"], start=1, stop=2)
        N(f"tm{i}", "mul", ["R1", f"bt{i}"]); N(f"tc{i}", "gmax", [f"tm{i}"])
        N(f"msb{i}", "mul", ["ms", f"bs{i}"])
        N(f"sm{i}", "mul", ["x0", f"msb{i}"]); N(f"sc{i}", "gmax", [f"sm{i}"])
        N(f"ex{i}", "gmax", [f"msb{i}"])
        N(f"pd{i}", "vmul", [f"tc{i}", f"sc{i}"])
        N(f"nm{i}", "dense", [f"pd{i}"], w="neg16", b="b1")
        N(f"mis{i}", "vmul", [f"ex{i}", f"nm{i}"])
        mis.append(f"mis{i}"); tcs.append(f"tc{i}"); msbs.append(f"msb{i}")
    N("mv", "vconcat", mis)
    L = np.tril(np.ones((NI, NI), f), -1)
    P["Wf"] = np.eye(NI, dtype=f) - L
    P["WL"] = L
    N("fl", "dense", ["mv"], w="Wf"); N("first", "vrelu", ["fl"])
    N("after", "dense", ["first"], w="WL")
    P["negN"] = -np.ones((1, NI), f)
    N("dl0", "dense", ["first"], w="negN", b="b1"); N("done", "vrelu", ["dl0"])
    cs = None; sk = None; lk = None
    for i in range(NI):
        N(f"fi{i}", "vslice", ["first"], start=i, stop=i + 1)
        N(f"ai{i}", "vslice", ["after"], start=i, stop=i + 1)
        N(f"ci{i}", "vmul", [f"fi{i}", tcs[i]])
        N(f"sk{i}", "mul", [msbs[i], f"fi{i}"])
        N(f"lk{i}", "mul", [msbs[i], f"ai{i}"])
        if cs is None:
            cs, sk, lk = f"ci{i}", f"sk{i}", f"lk{i}"
        else:
            cs = N(f"cs{i}", "vadd", [cs, f"ci{i}"])
            sk = N(f"ska{i}", "add", [sk, f"sk{i}"])
            lk = N(f"lka{i}", "add", [lk, f"lk{i}"])
    N("xc", "mul", ["x0", cs])
    P["W16c"] = np.ones((1, 16, 1, 1), f)
    N("cmap", "conv", ["xc"], w="W16c")
    N("pal2", "add", ["pal", "pal"]); N("srcw", "add", ["pal2", lk])
    N("srcm", "mul", ["cmap", "srcw"])
    Wr = np.zeros((1, 16, 7, 7), f)
    Wr[0, 0, 1, 1:7] = 1; Wr[0, 0, 6, 1:7] = 1
    P["Wring"] = Wr; P["bring"] = np.array([-11], f)
    N("rg0", "conv", ["x0"], w="Wring", b="bring"); N("rd", "relu", ["rg0"])
    P["W33"] = np.ones((1, 1, 3, 3), f)
    N("rdc", "conv", ["rd"], w="W33"); N("rdd", "min2", ["rdc", "one"])
    cm = np.ones(16, f); cm[0] = 0; cm[4] = 0; cm[5] = 0
    P["Dmask"] = np.diag(cm).astype(f)
    N("xs0", "mul", ["x0", "rdd"]); N("xsg", "gmax", ["xs0"])
    N("selc", "dense", ["xsg"], w="Dmask")
    N("sp", "vmul", ["selc", cs])
    P["ones16"] = np.ones((1, 16), f)
    N("selok", "dense", ["sp"], w="ones16")
    N("nsel", "dense", ["sp"], w="neg16", b="b1")
    N("ck1", "mul", [sk, "selok"]); N("ck2", "mul", ["srcm", "nsel"])
    N("click", "add", ["ck1", "ck2"])
    Wl = np.zeros((8, 1), f); Wl[5, 0] = 2
    bl = np.zeros(8, f); bl[6] = 1; bl[7] = -1
    P["Wl"] = Wl; P["bl"] = bl
    N("logits", "dense", ["done"], w="Wl", b="bl")
    return {"graph": G, "memory_size": 0, "params": P}

import numpy as np

def build():
    T = {
     "A1":"0010000100011100010011111","A2":"0000100101111111010010000",
     "A3":"1111110100100001010011111","A4":"0010011111101011010100100",
     "A5":"1000111111001001111110001","A6":"1111100100111001000011111",
     "A7":"1000111111100011000111011","B1":"1000011110100101001011111",
     "B2":"1111110001111011010111111","B3":"0011100101111111010011100",
     "B4":"0111001010111111000111111","B5":"1111110001100011111100100",
     "B6":"1111010011100011100101111","B7":"0010011111101011111100100",
     "C1":"1010110000101011000011111","C2":"1110110000000000000110111",
     "C3":"1011100100001000010011101","C4":"0010010101001001010100100",
     "C5":"1101110001101011000111011","C6":"1011110101111010000010101",
     "C7":"1000100000100011101101010"}
    names = [f + str(k) for f in "ABC" for k in range(1, 8)]
    base = [np.array([int(ch) for ch in T[n]], dtype=np.float64).reshape(5, 5) for n in names]
    Wt = np.zeros((84, 1, 7, 7)); bt = np.zeros(84)
    for r in range(4):
        for s in range(21):
            p = np.rot90(base[s], r)
            k = np.full((7, 7), -2.0)
            k[1:6, 1:6] = np.where(p > 0, 2.0, -2.0)
            Wt[r * 21 + s, 0] = k
            bt[r * 21 + s] = -(2 * p.sum() - 1)

    def shk(a, b):
        w = np.zeros((1, 1, 7, 7)); w[0, 0, a, b] = 1.0
        return w
    sr4 = np.zeros((4, 4, 3, 3)); sd4 = np.zeros((4, 4, 3, 3))
    for c in range(4):
        sr4[c, c, 1, 0] = 1.0
        sd4[c, c, 0, 1] = 1.0

    P = {
        "Wt": Wt, "bt": bt,
        "Wsum": np.ones((1, 21, 1, 1)),
        "Wsd3": shk(0, 3), "Wsu3": shk(6, 3), "Wcr3": shk(3, 6), "Wcr2": shk(3, 5),
        "Wcl1": shk(3, 4), "Wr3": shk(3, 0), "Wr1": shk(3, 2),
        "Wtm": np.array([1.0, -1.0, -1.0]).reshape(1, 3, 1, 1),
        "Wm": np.ones((1, 3, 1, 1)), "bm": np.array([-2.0]),
        "Wpri": np.array([-4032.0, -63.0, 5000.0]).reshape(1, 3, 1, 1),
        "Wcd": np.array([0.0, -1.0, 2.0]).reshape(1, 3, 1, 1),
        "Wrow": np.array([-1.0, 0.0, 2.0]).reshape(1, 3, 1, 1),
        "W63": np.array([63.0]).reshape(1, 1, 1, 1), "B1": np.array([1.0]),
        "Wsh4r": sr4, "Wsh4d": sd4,
        "Wna": np.array([[-1.0]]), "bna": np.array([1.0]),
        "Wc": np.zeros((1, 3, 1, 1)),
    }
    sets = [{4, 5, 6}, {1, 2, 3}, {0}]
    Wdec = np.zeros((63, 21))
    for b, st in enumerate(sets):
        for f in range(3):
            for i in range(7):
                for j in range(7):
                    if (j - i) % 7 in st:
                        Wdec[b * 21 + f * 7 + i, f * 7 + j] = 1.0
    Wout = np.zeros((8, 63))
    Wout[1, 0:21] = 1.0; Wout[2, 21:42] = 1.0; Wout[4, 42:63] = 1.0
    P["Wdec"] = Wdec; P["Wout"] = Wout
    P["bout"] = np.array([-5.0, 0.0, 0.05, -1.0, 0.1, -5.0, -5.0, -5.0])

    cnt = [0]
    def nid(t):
        cnt[0] += 1
        return "%s_%d" % (t, cnt[0])

    def E(L, op, ins, i=None, **kw):
        i = i or nid(op)
        d = {"id": i, "op": op, "in": ins}
        d.update(kw)
        L.append(d)
        return i

    def sh(L, x, d):
        while d > 0:
            s = 3 if d >= 3 else 1
            x = E(L, "conv", [x], w="Wr3" if s == 3 else "Wr1")
            d -= s
        return x

    def shl(L, x, d):
        while d > 0:
            s = 3 if d >= 3 else 1
            x = E(L, "conv", [x], w="Wcr3" if s == 3 else "Wcl1")
            d -= s
        return x

    g = []
    E(g, "slice", ["x0"], "is5", start=5, stop=6)
    E(g, "conv", ["is5"], "tc", w="Wt", b="bt")
    E(g, "relu", ["tc"], "tr")
    for q in range(4):
        E(g, "slice", ["tr"], "s%d" % q, start=21 * q, stop=21 * q + 21)
    E(g, "max2", ["s0", "s1"], "m01"); E(g, "max2", ["s2", "s3"], "m23")
    E(g, "max2", ["m01", "m23"], "S")
    E(g, "conv", ["S"], "anyS", w="Wsum")
    E(g, "slice", ["x0"], "z0", start=0, stop=1)
    E(g, "conv", ["z0"], "z3", w="Wsd3"); E(g, "conv", ["z3"], "z6", w="Wsd3")
    E(g, "conv", ["z0"], "u3", w="Wsu3"); E(g, "conv", ["u3"], "u6", w="Wsu3")
    E(g, "mul", ["z6", "u6"], "zz")
    E(g, "mul", ["zz", "anyS"], "sel")
    E(g, "rowmax", ["sel"], "bot")
    E(g, "slice", ["x0"], "c2", start=2, stop=3)
    E(g, "rowmax", ["c2"], "rule")
    E(g, "slice", ["x0"], "c3", start=3, stop=4)
    E(g, "conv", ["c3"], "cA", w="Wcr3"); E(g, "conv", ["cA"], "conn", w="Wcr2")
    E(g, "concat", ["anyS", "conn", "rule"], "a0i")
    E(g, "conv", ["a0i"], "a0c", w="Wm", b="bm"); E(g, "relu", ["a0c"], "A0")
    E(g, "slice", ["coords"], "one", start=2, stop=3)
    F = "anyS"
    for L in (1, 2, 3):
        F = sh(g, F, 7)
        nf = E(g, "sub", ["one", F])
        E(g, "mul", ["A0", nf], "A%d" % L)
    E(g, "conv", ["coords"], "pri", w="Wpri")
    E(g, "conv", ["coords"], "cdec", w="Wcd")
    E(g, "conv", ["coords"], "rdec", w="Wrow")
    E(g, "concat", ["anyS", "rule", "bot"], "tci")
    E(g, "conv", ["tci"], "tcc", w="Wtm"); E(g, "relu", ["tcc"], "TC")
    E(g, "mul", ["anyS", "bot"], "BC")

    def leftmost(X, out, dec="cdec"):
        v = E(g, "mul", [X, dec]); bv = E(g, "gmax", [v])
        dd = E(g, "sub", [v, bv]); cc = E(g, "conv", [dd], w="W63", b="B1")
        rr = E(g, "relu", [cc]); E(g, "mul", [X, rr], out)
    leftmost("TC", "p0"); leftmost("BC", "bp0")

    b = []
    q1 = sh(b, "p", 7); q2 = sh(b, q1, 7); q3 = sh(b, q2, 7)
    Gs = []
    for pm in ("p", q1, q2):
        u = E(b, "gmax", [E(b, "mul", ["S", pm])])
        Gs.append(E(b, "conv", [E(b, "mul", ["S", u])], w="Wsum"))
    G0a = sh(b, Gs[0], 7); G0b = sh(b, G0a, 7); G1a = sh(b, Gs[1], 7)
    mL1 = E(b, "mul", ["A1", Gs[0]])
    mL2 = E(b, "mul", [E(b, "mul", ["A2", Gs[1]]), G0a])
    mL3 = E(b, "mul", [E(b, "mul", [E(b, "mul", ["A3", Gs[2]]), G1a]), G0b])
    match = E(b, "add", [E(b, "add", [mL1, mL2]), mL3])
    sc = E(b, "mul", [match, "pri"]); best = E(b, "gmax", [sc])
    ad = E(b, "abs", [E(b, "sub", [sc, best])])
    rr = E(b, "relu", [E(b, "sub", ["one", ad])])
    asel = E(b, "mul", [match, rr])
    ls = [E(b, "gmax", [E(b, "mul", [asel, m])]) for m in (mL1, mL2, mL3)]
    pn = E(b, "add", [E(b, "add", [E(b, "mul", [q1, ls[0]]), E(b, "mul", [q2, ls[1]])]),
                      E(b, "mul", [q3, ls[2]])], "pn")
    rc0 = sh(b, asel, 10); rc1 = sh(b, rc0, 7); rc2 = sh(b, rc1, 7)
    rs = [E(b, "gmax", [E(b, "mul", ["S", rc])]) for rc in (rc0, rc1, rc2)]
    es = [E(b, "gmax", [E(b, "mul", ["anyS", rc])]) for rc in (rc0, rc1, rc2)]
    b1 = sh(b, "bp", 7); b2 = sh(b, b1, 7); b3 = sh(b, b2, 7)
    hs = [E(b, "gmax", [E(b, "mul", ["sel", bb])]) for bb in ("bp", b1, b2)]
    e01 = E(b, "vmul", [es[0], es[1]]); e012 = E(b, "vmul", [e01, es[2]])
    c1 = E(b, "vmul", [es[0], hs[1]]); c2 = E(b, "vmul", [e01, hs[2]])
    t = E(b, "vadd", ["tgt", E(b, "vmul", [rs[0], hs[0]])])
    t = E(b, "vadd", [t, E(b, "vmul", [rs[1], c1])])
    tn = E(b, "vadd", [t, E(b, "vmul", [rs[2], c2])], "tgtn")
    x1 = E(b, "mul", [b1, es[0]])
    x2 = E(b, "mul", [E(b, "sub", [b2, b1]), e01])
    x3 = E(b, "mul", [E(b, "sub", [b3, b2]), e012])
    bpn = E(b, "add", [E(b, "add", [x1, x2]), x3], "bpn")
    g.append({"id": "LP", "op": "loop", "T": 12,
              "state": [{"id": "p", "init": "p0"}, {"id": "bp", "init": "bp0"},
                        {"id": "tgt", "zeros": [21]}],
              "body": b, "next": {"p": pn, "bp": bpn, "tgt": tn}})

    # ---- alter_rules branch ----
    E(g, "mul", ["anyS", "rule"], "RT")
    E(g, "sub", ["anyS", "RT"], "low")
    leftmost("low", "Ttop", dec="rdec")
    E(g, "sub", ["low", "Ttop"], "Bm")
    E7 = shl(g, "RT", 7); E14 = shl(g, E7, 7)
    c7 = shl(g, "conn", 7); c14 = shl(g, c7, 7)
    t1 = E(g, "mul", [E7, c7])
    t2 = E(g, "mul", [E(g, "mul", [E7, E14]), c14])
    ssum = E(g, "add", [E(g, "add", ["conn", t1]), t2])
    E(g, "mul", ["RT", ssum], "Lm")
    E(g, "sub", ["RT", "Lm"], "Rm")
    E(g, "concat", ["Lm", "Rm", "Ttop", "Bm"], "X4")
    E(g, "rowsum", ["X4"], "rs4")
    g.append({"id": "PF", "op": "loop", "T": 63,
              "state": [{"id": "scs", "init": "X4"}, {"id": "srs", "init": "rs4"},
                        {"id": "acc", "zeros": [4, 64, 64]}],
              "body": [{"id": "scn", "op": "conv", "in": ["scs"], "w": "Wsh4r"},
                       {"id": "srn", "op": "conv", "in": ["srs"], "w": "Wsh4d"},
                       {"id": "acc1", "op": "add", "in": ["acc", "scn"]},
                       {"id": "accn", "op": "add", "in": ["acc1", "srn"]}],
              "next": {"scs": "scn", "srs": "srn", "acc": "accn"}})
    for i, nm in enumerate(["cL", "cR", "cT", "cB"]):
        E(g, "slice", ["PF.acc"], nm, start=i, stop=i + 1)
    E(g, "mul", ["sel", "RT"], "selR")
    leftmost("selR", "first")
    iL = E(g, "gmax", [E(g, "mul", ["first", "cL"])])
    iR = E(g, "gmax", [E(g, "mul", ["first", "cR"])])
    isL = E(g, "gmax", [E(g, "mul", ["first", "Lm"])])
    isR = E(g, "gmax", [E(g, "mul", ["first", "Rm"])])
    alt = E(g, "gmax", ["selR"], "alt")

    def pick(cmap, idx, rowm):
        d = E(g, "abs", [E(g, "sub", [cmap, idx])])
        eq = E(g, "relu", [E(g, "sub", ["one", d])])
        m = E(g, "mul", [eq, rowm])
        return E(g, "gmax", [E(g, "mul", ["S", m])])
    tL = pick("cT", iL, "Ttop"); tR = pick("cB", iR, "Bm")
    tgtA = E(g, "vadd", [E(g, "vmul", [tL, isL]), E(g, "vmul", [tR, isR])])
    curA = E(g, "gmax", [E(g, "mul", ["S", "first"])])
    na = E(g, "dense", [alt], w="Wna", b="bna")

    E(g, "mul", ["S", "sel"], "Sc"); E(g, "gmax", ["Sc"], "cur")
    tgtF = E(g, "vadd", [E(g, "vmul", ["LP.tgt", na]), E(g, "vmul", [tgtA, alt])])
    curF = E(g, "vadd", [E(g, "vmul", ["cur", na]), E(g, "vmul", [curA, alt])])
    E(g, "dense", [tgtF], "V", w="Wdec")
    E(g, "vconcat", [curF, curF, curF], "cur3")
    E(g, "vmul", ["V", "cur3"], "PV")
    E(g, "dense", ["PV"], "logits", w="Wout", b="bout")
    E(g, "conv", ["coords"], "click", w="Wc")
    params = {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 0, "params": params}

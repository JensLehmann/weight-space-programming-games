import numpy as np

def build():
    P = {}; G = []; cur = [G]; cnt = [0]
    def N(op, ins, **kw):
        cnt[0] += 1
        i = kw.pop("id", None) or f"n{cnt[0]}"
        d = {"id": i, "op": op, "in": ins}; d.update(kw); cur[0].append(d); return i
    def K(C, e):
        nm = f"K{C}_{e+3}"
        if nm not in P:
            w = np.zeros((C, C, 7, 7), np.float32)
            for k in range(C): w[k, k, 3, 3 + e] = 1.0
            P[nm] = w
        return nm
    def sh(x, d, C=1):
        while d != 0:
            e = max(-3, min(3, d)); x = N("conv", [x], w=K(C, e)); d -= e
        return x
    def sl(x, a): return N("slice", [x], start=a, stop=a + 1)
    def mul(a, b): return N("mul", [a, b])
    def add(a, b): return N("add", [a, b])
    def sub(a, b): return N("sub", [a, b])
    def inv(x): return N("conv", [x], w="Wneg", b="bone")
    def eq(X, s): return N("relu", [inv(N("abs", [N("sub", [X, s])]))])
    def eq2(X, s): return N("relu", [N("conv", [N("abs", [N("sub", [X, s])])], w="Wneg2", b="bone")])
    def gm(x): return N("gmax", [x])
    def ind(v):
        h = N("vrelu", [N("dense", [v], w="W6", b="b6")])
        return N("vrelu", [N("dense", [h], w="W3", b="b3")])
    def vs(v, k): return N("vslice", [v], start=k, stop=k + 1)

    W1 = np.zeros((13, 16, 7, 7), np.float32)
    reps = {}
    for di in range(-2, 3):
        for dj in range(-2, 3):
            r = []; a, b = di, dj
            for _ in range(4):
                r.append((a, b)); a, b = b, -a
            reps[(di, dj)] = min(r)
    uniq = sorted(set(reps.values())); oid = {u: k for k, u in enumerate(uniq)}
    for (di, dj), rep in reps.items():
        W1[0, 5, 3 + di, 3 + dj] = float(5 ** oid[rep])
    fam = 5.0 ** 7
    W1[0, 10, 6, 3] = fam; W1[0, 7, 6, 3] = 2 * fam; W1[0, 11, 6, 3] = 3 * fam
    for d in range(-2, 3):
        W1[1, 5, 1, 3 + d] = 1; W1[2, 5, 5, 3 + d] = 1
        W1[3, 5, 3 + d, 1] = 1; W1[4, 5, 3 + d, 5] = 1
    for i in range(7):
        for j in range(7):
            if max(abs(i - 3), abs(j - 3)) == 3: W1[5, 5, i, j] = 1
    W1[6, 5, 3, 4] = 1; W1[7, 5, 3, 2] = 1; W1[8, 5, 4, 3] = 1; W1[9, 5, 2, 3] = 1
    W1[10, 0, 6, 3] = 1; W1[11, 3, 3, 6] = 1; W1[12, 0, 0, 3] = 1
    P["W1"] = W1
    W2 = np.zeros((4, 13, 7, 7), np.float32)
    for k in range(7):
        W2[0, 6, k, 6] = 1; W2[0, 7, k, 0] = 1; W2[0, 8, 6, k] = 1; W2[0, 9, 0, k] = 1
    W2[1, 10, 6, 3] = 1; W2[2, 11, 3, 4] = 1; W2[3, 12, 0, 3] = 1
    P["W2"] = W2
    Wu = np.zeros((4, 13, 1, 1), np.float32)
    for k in range(4): Wu[k, 1 + k, 0, 0] = -1
    P["Wu"] = Wu; P["bu"] = np.ones(4, np.float32)
    Wc0 = np.zeros((1, 21, 1, 1), np.float32)
    Wc0[0, 0:4, 0, 0] = -1; Wc0[0, 9, 0, 0] = -1; Wc0[0, 17, 0, 0] = -1
    P["Wc0"] = Wc0
    P["bone"] = np.ones(1, np.float32)
    P["Wneg"] = -np.ones((1, 1, 1, 1), np.float32)
    P["Wneg2"] = -2 * np.ones((1, 1, 1, 1), np.float32)
    P["Wprio"] = np.array([-4032, -63, 5000], np.float32).reshape(1, 3, 1, 1)
    P["Wcol"] = np.array([0, -63, 100], np.float32).reshape(1, 3, 1, 1)
    P["Wrow"] = np.array([-63, 0, 100], np.float32).reshape(1, 3, 1, 1)
    P["WL"] = np.array([1, 2, 3], np.float32).reshape(1, 3, 1, 1)
    P["W11"] = np.ones((1, 2, 1, 1), np.float32)
    P["W6"] = np.array([[1], [-1], [1], [-1], [1], [-1]], np.float32)
    P["b6"] = np.array([-1, 1, -2, 2, -3, 3], np.float32)
    W3 = np.zeros((3, 6), np.float32)
    for k in range(3): W3[k, 2 * k] = -1; W3[k, 2 * k + 1] = -1
    P["W3"] = W3; P["b3"] = np.ones(3, np.float32)
    P["Wd"] = np.array([[1, -1], [-1, 1]], np.float32)
    P["Ws"] = np.array([[-1, -1]], np.float32)
    P["Wo7"] = np.array([[1, -7]], np.float32)
    Wdn = np.zeros((2, 2, 3, 3), np.float32); Wrt = np.zeros((2, 2, 3, 3), np.float32)
    for k in range(2): Wdn[k, k, 0, 1] = 1; Wrt[k, k, 1, 0] = 1
    P["Wdn"] = Wdn; P["Wrt"] = Wrt
    Wo = np.zeros((8, 1), np.float32); Wo[4, 0] = 2.0
    P["Wo"] = Wo
    P["bo"] = np.array([-10, -5, 1, -5, 0, -10, -10, -10], np.float32)

    F1 = N("conv", ["x0"], w="W1")
    F2 = N("conv", [F1], w="W2")
    U = N("relu", [N("conv", [F1], w="Wu", b="bu")])
    cen = N("relu", [N("conv", [N("concat", [U, F1, F2])], w="Wc0", b="bone")])
    S = sl(F1, 0); Z6 = sl(F2, 1); LCc = sl(F2, 2); Zm6 = sl(F2, 3)
    tc = mul(mul(cen, Z6), Zm6)
    top = N("rowmax", [sl("x0", 2)])
    PC = mul(cen, inv(top))
    trm = N("rowmax", [tc])
    tg = mul(PC, trm); sr = N("sub", [PC, tg])
    colp = N("conv", ["coords"], w="Wcol")
    def leftmost(C):
        v = mul(C, colp); m = gm(v)
        return mul(C, eq2(v, m))
    sp0 = leftmost(sr); tp0 = leftmost(tg)
    SC = mul(S, cen)
    LA = mul(mul(cen, top), LCc)
    X2 = N("concat", [SC, cen])
    L7 = sh(X2, -7, 2); L14 = sh(L7, -7, 2)
    R10 = sh(X2, 10, 2); R17 = sh(R10, 7, 2); R24 = sh(R17, 7, 2)
    A7 = sl(L7, 0); h7 = sl(L7, 1); A14 = sl(L14, 0); h14 = sl(L14, 1)
    I1 = mul(LA, inv(h7)); LAh = mul(LA, h7)
    I2 = mul(LAh, inv(h14)); I3 = mul(LAh, h14)
    Lmap = N("conv", [N("concat", [I1, I2, I3])], w="WL")
    R0 = sl(R10, 0); R1 = sl(R17, 0); h17 = sl(R17, 1)
    R2 = mul(sl(R24, 0), h17); h1724 = mul(h17, sl(R24, 1))
    RL = N("conv", [N("concat", [h17, h1724])], w="W11", b="bone")
    prio = N("conv", ["coords"], w="Wprio")

    body = []; cur[0] = body
    sp7 = sh("sp", -7); sp14 = sh(sp7, -7); sp21 = sh(sp14, -7)
    tp7 = sh("tp", -7); tp14 = sh(tp7, -7); tp21 = sh(tp14, -7)
    s0 = gm(mul("sp", SC)); s1 = gm(mul(sp7, SC)); s2 = gm(mul(sp14, SC))
    M1 = mul(I1, eq(SC, s0))
    M2 = mul(mul(I2, eq(A7, s0)), eq(SC, s1))
    M3 = mul(mul(mul(I3, eq(A14, s0)), eq(A7, s1)), eq(SC, s2))
    M = add(add(M1, M2), M3)
    MP = mul(M, prio); mx = gm(MP)
    sel = mul(M, eq2(MP, mx))
    Lv = gm(mul(sel, Lmap)); rl = gm(mul(sel, RL))
    r0 = gm(mul(sel, R0)); r1 = gm(mul(sel, R1)); r2 = gm(mul(sel, R2))
    iL = ind(Lv); iR = ind(rl)
    def pick(a, b, c, iv):
        return add(add(mul(a, vs(iv, 0)), mul(b, vs(iv, 1))), mul(c, vs(iv, 2)))
    spn = pick(sp7, sp14, sp21, iL); tpn = pick(tp7, tp14, tp21, iR)
    Qn = add(add(add("Q", mul("tp", r0)), mul(tp7, r1)), mul(tp14, r2))
    cur[0] = G
    G.append({"id": "LP", "op": "loop", "T": 12,
              "state": [{"id": "sp", "init": sp0}, {"id": "tp", "init": tp0},
                        {"id": "Q", "zeros": [1, 64, 64]}],
              "body": body, "next": {"sp": spn, "tp": tpn, "Q": Qn}})
    dv0 = gm(mul("LP.Q", tc))
    tv = gm(mul(tc, S))

    # ---- alter-rules mode ----
    rowp = N("conv", ["coords"], w="Wrow")
    vr = mul(PC, rowp); srcRow = mul(PC, eq2(vr, gm(vr))); tgtRow = sub(PC, srcRow)
    mS = gm(mul(srcRow, colp)); mT = gm(mul(tgtRow, colp))
    X = N("concat", [Lmap, mul(RL, LA)])
    XR = N("rowsum", [X])
    body2 = []; cur[0] = body2
    an = add(XR, N("conv", ["pa"], w="Wdn"))
    bn = add(X, N("conv", ["pb"], w="Wrt"))
    cur[0] = G
    G.append({"id": "LP2", "op": "loop", "T": 64,
              "state": [{"id": "pa", "zeros": [2, 64, 64]}, {"id": "pb", "zeros": [2, 64, 64]}],
              "body": body2, "next": {"pa": an, "pb": bn}})
    OFF = sub(add("LP2.pa", "LP2.pb"), add(XR, X))
    OFFs = sl(OFF, 0); OFFt = mul(sl(OFF, 1), LA)
    LFmask = add(add(I1, sh(I2, 7)), sh(I3, 14))
    LFoff = add(add(mul(OFFs, I1), sh(mul(OFFs, I2), 7)), sh(mul(OFFs, I3), 14))
    RFmask = sh(LA, -10); RFoff = sh(OFFt, -10)
    isL = gm(mul(tc, LFmask)); offL = gm(mul(tc, LFoff))
    isR = gm(mul(tc, RFmask)); offR = gm(mul(tc, RFoff))
    cS = N("dense", [N("vconcat", [mS, offL])], w="Wo7")
    cT = N("dense", [N("vconcat", [mT, offR])], w="Wo7")
    dS = gm(mul(mul(srcRow, SC), eq(colp, cS)))
    dT = gm(mul(mul(tgtRow, SC), eq(colp, cT)))
    dv = N("vadd", [N("vadd", [dv0, N("vmul", [isL, dS])]), N("vmul", [isR, dT])])

    td = N("vconcat", [tv, dv])
    ddr = N("vrelu", [N("dense", [td], w="Wd")])
    eqv = N("vrelu", [N("dense", [ddr], w="Ws", b="bone")])
    N("dense", [eqv], w="Wo", b="bo", id="logits")
    N("slice", [cen], start=0, stop=1, id="click")
    return {"graph": G, "memory_size": 0, "params": P}

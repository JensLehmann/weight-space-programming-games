import numpy as np

def build():
    P = {}
    G = []
    cur = [G]
    cnt = [0]
    def N(op, ins, idn=None, **f):
        cnt[0] += 1
        i = idn or ("n%d" % cnt[0])
        d = {"id": i, "op": op, "in": list(ins)}
        d.update(f)
        cur[0].append(d)
        return i
    def prm(name, arr):
        P[name] = np.asarray(arr, np.float32)
        return name

    cells = [(i, j) for i in range(10) for j in range(10) if i != j and i + j != 9]
    pts = [(3 + i, 3 + j) for i, j in cells] + [(34 + i, 27 + j) for i, j in cells]
    bprobe = [(28, 31), (29, 41), (39, 42), (48, 41), (49, 31), (48, 22), (39, 21), (29, 22)]
    sprobe = [(19, 31), (38, 51), (58, 31), (38, 12)]
    pts += bprobe + sprobe
    NP = len(pts)
    Wr = np.zeros((NP, 3, 1, 1)); Wc = np.zeros((NP, 3, 1, 1))
    for k, (r, c) in enumerate(pts):
        Wr[k, 0, 0, 0] = 63; Wr[k, 2, 0, 0] = -r
        Wc[k, 1, 0, 0] = 63; Wc[k, 2, 0, 0] = -c
    dr = N("conv", ["coords"], w=prm("Wr", Wr))
    dc = N("conv", ["coords"], w=prm("Wc", Wc))
    s_ = N("add", [N("abs", [dr]), N("abs", [dc])])
    one_sp = N("slice", ["coords"], start=2, stop=3)
    M = N("relu", [N("sub", [one_sp, s_])])
    v = N("conv", ["x0"], w=prm("Wv", np.arange(16).reshape(1, 16, 1, 1)))
    g = N("gmax", [N("mul", [M, v])])
    Ms = N("slice", [M], start=168, stop=172)

    cv = [1.0, -1.0, 40.0] + [40 * (0.5 - c) for c in range(1, 16)] + [float(c) for c in range(1, 16)] + [79.5]
    cst = N("dense", ["avail"], w=prm("Wk", np.zeros((len(cv), 8))), b=prm("bk", cv))
    K = lambda i: N("vslice", [cst], start=i, stop=i + 1)
    one, m1, k40 = K(0), K(1), K(2)
    off = {c: K(2 + c) for c in range(1, 16)}
    cval = {c: K(17 + c) for c in range(1, 16)}
    c795 = K(33)
    neg = lambda a: N("vmul", [a, m1])
    omin = lambda a: N("vadd", [neg(a), one])
    def vabs(a):
        return N("vadd", [N("vrelu", [a]), N("vrelu", [neg(a)])])

    Gs = N("vmul", [g, k40])
    step = {c: N("vsigmoid", [N("vadd", [Gs, off[c]])]) for c in range(1, 16)}
    oh = {0: omin(step[1]), 15: step[15]}
    for c in range(1, 15):
        oh[c] = N("vadd", [step[c], neg(step[c + 1])])
    T = {c: N("vslice", [oh[c]], start=0, stop=80) for c in range(16)}
    Cn = {c: N("vslice", [oh[c]], start=80, stop=160) for c in range(16)}
    Bp = {c: N("vslice", [oh[c]], start=160, stop=168) for c in range(16)}
    S0 = N("vslice", [oh[0]], start=168, stop=172)
    corr = None
    for c in range(16):
        t = N("vmul", [T[c], Cn[c]])
        corr = t if corr is None else N("vadd", [corr, t])
    W = omin(corr)
    bvec = omin(Bp[5])
    ones8 = prm("ones8", np.ones((1, 8)))
    sl = [N("dense", [N("vmul", [bvec, Bp[c]])], w=ones8) for c in range(16)]
    svec = N("vconcat", sl)
    sval = N("dense", [svec], w=prm("cidx", np.arange(16).reshape(1, 16)))
    fnow = N("dense", [S0], w=prm("ones4", np.ones((1, 4))))
    flag = N("vadd", [fnow, N("vrelu", [N("vadd", ["mem", neg(fnow)])])], idn="mem_next")
    gate = N("dense", [flag], w=prm("Gw", [[0]] * 8 + [[1]] * 4), b=prm("Gb", [1] * 8 + [0] * 4))

    def region(k, i, j):
        return [i < 5, j >= i, j >= 5, j >= 9 - i, i >= 5, j <= i, j < 5, j <= 9 - i,
                i < 3 and 3 <= j < 7, 3 <= i < 7 and j >= 7,
                i >= 7 and 3 <= j < 7, 3 <= i < 7 and j < 3][k]
    R = np.array([[1.0 if region(k, i, j) else 0.0 for (i, j) in cells] for k in range(12)])
    Rn = prm("R", R); RTn = prm("RT", R.T)

    body = []
    cur[0] = body
    u = omin("E")
    tot = N("dense", [u], w=Rn)
    nn = neg(N("vrelu", [N("vadd", [tot, m1])]))
    Ps = None; CV = None
    for c in range(16):
        cn = N("dense", [N("vmul", [u, T[c]])], w=Rn)
        p = N("vrelu", [N("vadd", [cn, nn])])
        Ps = p if Ps is None else N("vadd", [Ps, p])
        if c > 0:
            q = N("vmul", [p, cval[c]])
            CV = q if CV is None else N("vadd", [CV, q])
    Pg = N("vmul", [Ps, gate]); CVg = N("vmul", [CV, gate])
    wc = N("dense", [N("vmul", [u, W])], w=Rn)
    wp = N("vadd", [wc, neg(N("vrelu", [N("vadd", [wc, m1])]))])
    n1 = N("vadd", ["n", one])
    sc2 = N("vadd", ["sc", N("vmul", [N("vmul", [Pg, wp]), n1])])
    ca2 = N("vadd", ["ca", CVg])
    ex = N("vadd", ["E", N("dense", [Pg], w=RTn)])
    E2 = N("vadd", [ex, neg(N("vrelu", [N("vadd", [ex, m1])]))])
    cur[0] = G
    G.append({"id": "L", "op": "loop", "T": 12,
              "state": [{"id": "E", "zeros": [80]}, {"id": "n", "zeros": [1]},
                        {"id": "sc", "zeros": [12]}, {"id": "ca", "zeros": [12]}],
              "body": body, "next": {"E": E2, "n": n1, "sc": sc2, "ca": ca2}})
    SC, CA = "L.sc", "L.ca"

    # incomplete explanation indicator
    sumE = N("dense", ["L.E"], w=prm("ones80", np.ones((1, 80))))
    incx = N("vmul", [N("vrelu", [N("vadd", [neg(sumE), c795])]), k40])
    inc = N("vadd", [incx, neg(N("vrelu", [N("vadd", [incx, m1])]))])

    cm = N("vrelu", [omin(vabs(N("vadd", [CA, neg(sval)])))])
    opos = [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6]
    Bm = np.zeros((12, 8))
    for k in range(12):
        Bm[k, opos[k]] = 1
    pmt = N("dense", [bvec], w=prm("Bm", Bm))
    I12 = np.eye(12)
    sco = N("dense", [N("vconcat", [SC, cm, pmt])], w=prm("Wsc", np.hstack([32 * I12, 4 * I12, 4 * I12])),
            b=prm("bsc", 0.25 * np.arange(12)))
    Dw = []; Sm = np.zeros((12, 132))
    for k in range(12):
        for j in range(12):
            if j != k:
                r = np.zeros(12); r[k] = 1; r[j] = -1
                Sm[k, len(Dw)] = 1
                Dw.append(r)
    D = N("dense", [sco], w=prm("Dw", np.array(Dw)))
    st = N("vsigmoid", [N("vmul", [D, k40])])
    sel = N("vsigmoid", [N("vmul", [N("dense", [st], w=prm("Sm", Sm), b=prm("bsm", [-10.5] * 12)), k40])])
    cstar = N("dense", [N("vmul", [sel, CA])], w=prm("ones12", np.ones((1, 12))))
    a = vabs(N("vadd", [cstar, neg(sval)]))
    nc = N("vadd", [a, neg(N("vrelu", [N("vadd", [a, m1])]))])
    g1 = omin(nc)
    tpos = N("dense", [sel], w=prm("BmT", Bm.T))
    atpos = N("dense", [N("vmul", [tpos, bvec])], w=ones8)
    isstrip = N("dense", [sel], w=prm("wis", [[0] * 8 + [1] * 4]))
    Eb = np.zeros((64, 8)); Et = np.zeros((64, 8)); Mv = np.zeros((4, 64))
    cw = [4, 2, 2, 3, 3, 1, 1, 4]; ccw = [3, 3, 1, 1, 4, 4, 2, 2]
    for b in range(8):
        for t in range(8):
            i = b * 8 + t
            Eb[i, b] = 1; Et[i, t] = 1
            if b != t:
                d = (t - b) % 8
                act = cw[b] if d <= 4 else ccw[b]
                Mv[act - 1, i] = 1
    pr = N("vmul", [N("dense", [bvec], w=prm("Eb", Eb)), N("dense", [tpos], w=prm("Et", Et))])
    mv = N("dense", [pr], w=prm("Mv", Mv))
    mvg = N("vmul", [mv, N("vmul", [g1, omin(atpos)])])
    ga = N("vmul", [g1, atpos])
    cl = N("vadd", [nc, N("vmul", [ga, isstrip])])
    a5 = N("vmul", [ga, omin(isstrip)])

    oddp = N("dense", [bvec], w=prm("oddw", [[b % 2 for b in range(8)]]))
    expl = N("vmul", [N("vmul", [oddp, omin(flag)]), inc])
    keep = omin(expl)
    Ex = np.zeros((4, 8))
    for b in (1, 3, 5, 7):
        Ex[cw[b] - 1, b] = 1
    mvx = N("vmul", [N("dense", [bvec], w=prm("Ex", Ex)), expl])
    mvf = N("vadd", [N("vmul", [mvg, keep]), mvx])
    clk = N("vmul", [cl, keep])
    a5k = N("vmul", [a5, keep])

    Wl = np.zeros((8, 6))
    Wl[6, 0] = 10; Wl[5, 1] = 10
    for i in range(4):
        Wl[1 + i, 2 + i] = 10
    N("dense", [N("vconcat", [clk, a5k, mvf])], idn="logits", w=prm("Wl", Wl),
      b=prm("bl", [-5, 0, 0, 0, 0, 0, 0, -5]))

    cneg = N("dense", ["avail"], w=prm("Wk2", np.zeros((16, 8))), b=prm("bk2", -np.arange(16)))
    want = N("vrelu", [omin(vabs(N("vadd", [cneg, cstar])))])
    pal = N("conv", [N("mul", ["x0", want])], w=prm("ones16", np.ones((1, 16, 1, 1))))
    Wb = np.zeros((2, 3, 1, 1)); Wb[0, 0] = 63; Wb[1, 1] = 40 * 63
    bc = N("conv", ["coords"], w=prm("Wb", Wb), b=prm("bb", [-4.0, -40 * 18.5]))
    br = N("relu", [N("conv", [N("abs", [N("slice", [bc], start=0, stop=1)])],
                      w=prm("Wbr", [[[[-1.0]]]]), b=prm("bbr", [1.5]))])
    bcol = N("sigmoid", [N("slice", [bc], start=1, stop=2)])
    palt = N("mul", [N("mul", [pal, N("mul", [br, bcol])]), nc])
    sw = N("vmul", [N("vslice", [sel], start=8, stop=12), ga])
    stc = N("conv", [N("mul", [Ms, sw])], w=prm("ones4s", np.ones((1, 4, 1, 1))))
    N("add", [palt, stc], idn="click")

    return {"graph": G, "memory_size": 1, "params": P}

import numpy as np

def build():
    G = []; P = {}; cnt = [0]
    def nid(p):
        cnt[0] += 1; return "%s%d" % (p, cnt[0])
    def node(op, ins, L=None, id=None, **kw):
        L = G if L is None else L
        i = id or nid(op)
        d = {"id": i, "op": op, "in": ins}; d.update(kw); L.append(d); return i
    def par(a):
        n = nid("P"); P[n] = np.asarray(a, np.float32); return n
    def conv(x, W, b=None, L=None, id=None):
        kw = {"w": par(W)}
        if b is not None: kw["b"] = par(b)
        return node("conv", [x], L, id, **kw)
    def dense(x, W, b=None, L=None, id=None):
        kw = {"w": par(W)}
        if b is not None: kw["b"] = par(b)
        return node("dense", [x], L, id, **kw)
    def eq(x, W, b, L=None):
        W = np.asarray(W, float); b = np.asarray(b, float)
        a1 = node("vrelu", [dense(x, -W, 1 - b, L)], L)
        a2 = node("vrelu", [dense(x, W, 1 + b, L)], L)
        return node("vmul", [a1, a2], L)

    # ---- atoms ----
    atoms = [('T',1,1,1,(1,4)),('T',1,0,1,(1,5)),('T',1,1,0,(3,4)),('T',1,0,0,(3,5)),
             ('R',1,0,1,(4,7)),('R',0,0,1,(5,7)),('R',1,0,0,(2,8)),('R',0,0,0,(7,8)),
             ('B',0,1,1,(8,4)),('B',0,0,1,(8,5)),('B',0,1,0,(6,4)),('B',0,0,0,(6,5)),
             ('L',1,1,1,(4,2)),('L',0,1,1,(5,2)),('L',1,1,0,(2,1)),('L',0,1,0,(7,1))]
    A = np.zeros((12, 16))
    for a, (t, top, left, st, _) in enumerate(atoms):
        mem = [top, t in 'TR', not left, t in 'RB', not top, t in 'BL', left, t in 'TL',
               st and t == 'T', st and t == 'R', st and t == 'B', st and t == 'L']
        for o in range(12): A[o, a] = 1 if mem[o] else 0
    cells = [(3+i, 3+j) for (_,_,_,_,(i,j)) in atoms] + [(34+i, 27+j) for (_,_,_,_,(i,j)) in atoms]
    cells += [(24,31),(21,39),(38,46),(46,49),(53,31),(56,24),(38,17),(31,14)]
    cells += [(18,31),(38,52),(59,31),(38,11)]
    K = len(cells); N = 2*K + 2

    # ---- spatial perception ----
    Wrc = np.zeros((N, 3, 1, 1))
    for k, (r, c) in enumerate(cells):
        Wrc[k, :, 0, 0] = [63, 0, -r]; Wrc[K+k, :, 0, 0] = [0, 63, -c]
    Wrc[2*K, :, 0, 0] = [63, 0, -4]; Wrc[2*K+1, :, 0, 0] = [63, 0, -7]
    rc = conv("coords", Wrc)
    ab = node("abs", [rc])
    I_N = np.eye(N)[:, :, None, None]
    b1 = node("relu", [conv(ab, 6*I_N, -2*np.ones(N))])
    ind = node("relu", [conv(b1, -I_N, np.ones(N))])
    RI = node("slice", [ind], start=0, stop=K)
    CI = node("slice", [ind], start=K, stop=2*K)
    r4 = node("slice", [ind], start=2*K, stop=2*K+1)
    r7 = node("slice", [ind], start=2*K+1, stop=2*K+2)
    CELL = node("mul", [RI, CI])
    cg = node("relu", [conv("coords", np.array([[0,63,-17],[0,63,-18]], float)[:, :, None, None])])
    cge = conv(cg, np.array([[1, -1]], float)[:, :, None, None])
    CV = conv("x0", np.arange(16, dtype=float).reshape(1, 16, 1, 1))
    vals = node("gmax", [node("mul", [CV, CELL])])
    m7 = node("mul", [r7, cge]); m4 = node("mul", [r4, cge])
    x0b = node("slice", ["x0"], start=0, stop=1)
    cmk = node("colmax", [node("mul", [x0b, m7])])
    sel4 = node("mul", [m4, cmk])
    curr = node("gmax", [node("mul", ["x0", sel4])])
    D = np.eye(16); D[4, 4] = 0
    cur = dense(curr, D)

    # ---- vector features ----
    tgt = node("vslice", [vals], start=0, stop=16)
    REP = np.zeros((256, 16)); cvb = np.zeros(256)
    for a in range(16):
        for c in range(16): REP[a*16+c, a] = 1; cvb[a*16+c] = -c
    T1h = eq(tgt, REP, cvb)
    corr = eq(node("vslice", [vals], start=0, stop=32), np.hstack([np.eye(16), -np.eye(16)]), np.zeros(16))
    q = eq(node("vslice", [vals], start=32, stop=40), np.eye(8), -2*np.ones(8))
    cupv = eq(node("vslice", [vals], start=40, stop=44), np.eye(4), -2*np.ones(4))
    xm = node("vadd", ["mem", dense(cupv, np.ones((1, 4)))])
    ym = node("vrelu", [dense(xm, [[1.0]], [-1.0])])
    memn = dense(node("vconcat", [xm, ym]), [[1.0, -1.0]], id="mem_next")
    gate = dense(memn, np.array([[0]*8 + [1]*4], float).T, np.array([1]*8 + [0]*4, float))
    wrong = dense(corr, -np.eye(16), np.ones(16))

    # ---- planner loop ----
    B = []
    r = dense("lp_p", -np.eye(16), np.ones(16), B)
    rw = node("vmul", [r, wrong], B)
    rT = node("vmul", [dense(r, REP, None, B), T1h], B)
    no = dense(r, A, None, B)
    empt = node("vrelu", [dense(no, -np.eye(12), np.ones(12), B)], B)
    vin = node("vconcat", [rT, r, empt], B)
    Wv = np.zeros((192, 284))
    for o in range(12):
        for c in range(16):
            row = o*16 + c
            for a in range(16):
                Wv[row, a*16+c] = A[o, a]; Wv[row, 256+a] = -A[o, a]
            Wv[row, 272+o] = -1
    voc = node("vrelu", [dense(vin, Wv, np.ones(192), B)], B)
    Sc = np.zeros((12, 192)); WE = np.zeros((192, 12)); Co = np.zeros((16, 192))
    for o in range(12):
        for c in range(16):
            Sc[o, o*16+c] = 1; WE[o*16+c, o] = 1; Co[c, o*16+c] = 1
    vopg = node("vmul", [dense(voc, Sc, None, B), gate], B)
    raw = dense(node("vconcat", [rw, r], B), np.hstack([256*A, 16*A]),
                np.array([15 - o for o in range(12)], float), B)
    s = node("vmul", [vopg, raw], B)
    Wd = np.zeros((288, 12)); bd = np.zeros(288); Wl = np.zeros((12, 288))
    for o in range(12):
        for j in range(12):
            k = o*12 + j
            if j != o:
                Wd[k, j] += 1; Wd[k, o] -= 1; Wd[144+k, j] += 1; Wd[144+k, o] -= 1
            bd[144+k] = -1
            Wl[o, k] = 1; Wl[o, 144+k] = -1
    lose = dense(node("vrelu", [dense(s, Wd, bd, B)], B), Wl, None, B)
    w = node("vrelu", [dense(node("vconcat", [vopg, lose], B), np.hstack([np.eye(12), -np.eye(12)]), None, B)], B)
    cover = dense(w, A.T, None, B)
    badt = dense(rw, np.ones((1, 16)), None, B)
    q1 = node("vrelu", [dense(badt, [[1.0]], [-1.0], B)], B)
    go = dense(node("vconcat", [badt, q1], B), [[1.0, -1.0]], None, B)
    pn = node("vadd", ["lp_p", node("vmul", [node("vmul", [r, cover], B), go], B)], B)
    cw = dense(node("vmul", [dense(w, WE, None, B), voc], B), Co, None, B)
    dro = dense(node("vconcat", [w, "lp_ro"], B), np.hstack([np.eye(12), -np.eye(12)]), None, B)
    ron = node("vadd", ["lp_ro", node("vmul", [dro, go], B)], B)
    drc = dense(node("vconcat", [cw, "lp_rc"], B), np.hstack([np.eye(16), -np.eye(16)]), None, B)
    rcn = node("vadd", ["lp_rc", node("vmul", [drc, go], B)], B)
    G.append({"id": "PL", "op": "loop", "T": 17,
              "state": [{"id": "lp_p", "zeros": [16]}, {"id": "lp_ro", "zeros": [12]}, {"id": "lp_rc", "zeros": [16]}],
              "body": B, "next": {"lp_p": pn, "lp_ro": ron, "lp_rc": rcn}})
    os_, cs = "PL.lp_ro", "PL.lp_rc"

    # ---- action logic ----
    plan = dense(cs, np.ones((1, 16)))
    match = dense(node("vmul", [cs, cur]), np.ones((1, 16)))
    M = np.zeros((8, 12))
    for o in range(12): M[o if o < 8 else 2*(o-8), o] = 1
    pv = dense(os_, M)
    Qd = np.zeros((64, 8)); Pd = np.zeros((64, 8)); MT = np.zeros((8, 64)); DG = np.zeros((1, 64))
    CWa = [4, 2, 2, 3, 3, 1, 1, 4]; CCWa = [3, 3, 1, 1, 4, 4, 2, 2]
    for qq in range(8):
        for pp in range(8):
            k = qq*8 + pp; Qd[k, qq] = 1; Pd[k, pp] = 1
            d = (pp - qq) % 8
            if d == 0: DG[0, k] = 1
            elif d <= 4: MT[CWa[qq], k] = 1
            else: MT[CCWa[qq], k] = 1
    pair = node("vmul", [dense(q, Qd), dense(pv, Pd)])
    at = dense(pair, DG); mv = dense(pair, MT)
    cupf = dense(os_, np.array([[0]*8 + [1]*4], float))
    CWm = np.zeros((8, 8))
    for qq in range(8): CWm[CWa[qq], qq] = 1
    cwfb = dense(q, CWm)
    F = node("vconcat", [plan, match, at, cupf, mv, cwfb])
    Wh = np.zeros((19, 20)); bh = np.zeros(19)
    Wh[0, 0] = 1; Wh[0, 1] = -1
    Wh[1, 1] = 1; Wh[1, 2] = 1; Wh[1, 3] = 1; bh[1] = -2
    Wh[2, 1] = 1; Wh[2, 2] = 1; Wh[2, 3] = -1; bh[2] = -1
    for a in range(8):
        Wh[3+a, 4+a] = 1; Wh[3+a, 1] = 1; bh[3+a] = -1
        Wh[11+a, 12+a] = 1; Wh[11+a, 0] = -1
    hr = node("vrelu", [dense(F, Wh, bh)])
    Wo = np.zeros((8, 19)); bo = np.zeros(8)
    Wo[6, 0] = 10; Wo[6, 1] = 10; Wo[5, 2] = 10
    for a in range(8): Wo[a, 3+a] += 10; Wo[a, 11+a] += 10
    bo[1] = 0.5
    dense(hr, Wo, bo, id="logits")

    # ---- click map ----
    pms = conv(node("mul", ["x0", cs]), 2*np.ones((1, 16, 1, 1)))
    pmap = node("mul", [pms, m4])
    cupsel = node("vslice", [os_], start=8, stop=12)
    cupcells = node("slice", [CELL], start=K-4, stop=K)
    cmap = conv(node("mul", [cupcells, cupsel]), 4*np.ones((1, 4, 1, 1)))
    cmapg = node("mul", [cmap, match])
    node("add", [pmap, cmapg], id="click")
    return {"graph": G, "memory_size": 1, "params": P}

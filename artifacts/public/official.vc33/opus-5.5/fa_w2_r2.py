import numpy as np

def build():
    G = []; P = {}; cnt = [0]

    def node(op, ins, nid=None, **kw):
        cnt[0] += 1
        i = nid or "n%d_%s" % (cnt[0], op)
        d = {"id": i, "op": op, "in": ins}; d.update(kw); G.append(d); return i

    def par(a):
        n = "p%d" % len(P); P[n] = np.asarray(a, np.float32); return n

    def conv1(x, W, b=None, nid=None):
        W = np.asarray(W, np.float32); W = W.reshape(W.shape[0], W.shape[1], 1, 1)
        kw = {"w": par(W)}
        if b is not None: kw["b"] = par(b)
        return node("conv", [x], nid, **kw)

    def dense(x, W, b=None, nid=None):
        kw = {"w": par(np.asarray(W, np.float32))}
        if b is not None: kw["b"] = par(b)
        return node("dense", [x], nid, **kw)

    def clip_s(x, wv, b):
        wv = np.asarray(wv, np.float32)
        h = conv1(x, np.stack([wv, wv]), [b, b - 1.0])
        return conv1(node("relu", [h]), [[1.0, -1.0]])

    def clip_v(x, wv, b):
        wv = np.asarray(wv, np.float32)
        h = dense(x, np.stack([wv, wv]), [b, b - 1.0])
        return dense(node("vrelu", [h]), [[1.0, -1.0]])

    wc = np.zeros((1, 16)); wc[0, 11] = 1; wc[0, 14] = 1
    C = conv1("x0", wc)
    rowm = clip_s("coords", [126.0, 0, 0], -1.0)
    s4 = node("slice", ["x0"], start=4, stop=5)
    F = node("mul", [s4, rowm])
    N4 = node("conv", [F], w=par(np.ones((1, 1, 5, 5), np.float32)))
    N4c = clip_s(N4, [1.0], 0.0)
    B = node("mul", [C, N4c])
    T = node("sub", [C, B])

    Rc = conv1("coords", [[63, 0, 0], [-63, 0, 63], [0, 63, 0], [0, -63, 63]])
    gB = node("gmax", [node("mul", [Rc, B])])
    gF = node("gmax", [node("mul", [Rc, F])])
    gT = node("gmax", [node("mul", [Rc, T])])
    gall = node("vconcat", [gB, gF, gT])

    W = np.zeros((8, 12)); bb = np.zeros(8)
    W[0, [4, 5, 0, 1]] = [1, -1, -1, 1]
    W[1, [6, 7, 2, 3]] = [1, -1, -1, 1]
    W[2, [8, 9, 0, 1]] = [1, -1, -1, 1]
    W[3, [10, 11, 2, 3]] = [1, -1, -1, 1]
    W[4, [0, 1, 2, 3]] = [1, 1, -1, -1]
    W[5, [0, 1]] = [0.5, -0.5]; bb[5] = 31.5
    W[6, [2, 3]] = [0.5, -0.5]; bb[6] = 31.5
    W[7, [0, 1, 2, 3]] = [1, 2, 3, 5]
    Dd = dense(gall, W, bb)

    e4 = np.zeros(8); e4[4] = 1
    h = clip_v(Dd, e4, 0.0)
    prpc = node("vmul", [node("vslice", [Dd], start=0, stop=2),
                         node("vslice", [Dd], start=2, stop=4)])
    q = dense(prpc, [[-1.0, 1.0]])
    pr = dense(prpc, [[1.0, 0.0]])
    s = node("vadd", [pr, node("vmul", [h, q])])

    P2 = conv1("coords", [[63, 0, 0], [0, 63, 0]])
    cvec = node("vslice", [Dd], start=5, stop=7)
    D2 = node("abs", [node("sub", [P2, cvec])])
    wv = dense(h, [[1.0], [-1.0]], [0.0, 1.0])
    DW = node("mul", [D2, wv])
    Z = conv1(DW, [[-1.0, -1.0]], [200.0])
    wv2 = dense(h, [[-1.0], [1.0]], [1.0, 0.0])
    Dpar = conv1(node("mul", [D2, wv2]), [[1.0, 1.0]])

    Wt = np.zeros((2, 12)); Wt[0, 8] = .5; Wt[0, 9] = -.5; Wt[1, 10] = .5; Wt[1, 11] = -.5
    tvec = dense(gall, Wt, [31.5, 31.5])
    a1 = node("sub", [P2, cvec])
    a2 = node("sub", [tvec, P2])
    QW = node("mul", [node("mul", [a1, a2]), wv])
    qs = conv1(QW, [[1.0, 1.0]])
    betw = clip_s(qs, [1.0], 0.0)
    wg = np.zeros((1, 16)); wg[0, 1] = 1; wg[0, 12] = 1
    G1 = conv1("x0", wg)
    Gn = node("mul", [G1, betw])
    mg = node("gmax", [node("mul", [Gn, Z])])
    kg = clip_s(node("sub", [Z, mg]), [1.0], 3.0)
    Gs = node("mul", [Gn, kg])
    hasG = node("gmax", [Gs])
    gG = node("gmax", [node("mul", [Rc, Gs])])
    du = node("vslice", [Dd], start=0, stop=2)
    dd4 = node("vrelu", [dense(du, [[1, 0], [-1, 0], [0, 1], [0, -1]])])
    We = np.zeros((4, 8))
    We[0, 1] = 1; We[0, 5] = -1
    We[1, 0] = 1; We[1, 4] = -1
    We[2, 3] = 1; We[2, 7] = -1
    We[3, 2] = 1; We[3, 6] = -1
    ev = dense(node("vconcat", [gB, gG]), We)
    prod = node("vmul", [dd4, ev])
    sgrc = dense(prod, [[1, 1, 0, 0], [0, 0, 1, 1]])
    qg = dense(sgrc, [[-1.0, 1.0]])
    sgr = dense(sgrc, [[1.0, 0.0]])
    sg = node("vadd", [sgr, node("vmul", [h, qg])])
    diff = node("vadd", [sg, dense(s, [[-1.0]])])
    sf = node("vadd", [s, node("vmul", [hasG, diff])])
    grow = clip_v(sf, [1.0], 0.0)

    absf = dense(node("vrelu", [dense(sf, [[1.0], [-1.0]])]), [[1.0, 1.0]])
    e1v = node("vrelu", [dense(absf, [[-1.0]], [1.0])])
    eq = node("vmul", [e1v, hasG])

    V = node("vconcat", [Dd, "mem"])
    Wd = np.zeros((2, 10)); Wd[0, 7] = 1; Wd[0, 8] = -1; Wd[1, 7] = -1; Wd[1, 8] = 1
    ad = node("vrelu", [dense(V, Wd)])
    unch = clip_v(ad, [-1.0, -1.0], 1.0)
    fp = node("vslice", ["mem"], start=1, stop=2)
    far = node("vrelu", [dense(node("vconcat", [unch, fp, grow]), [[1.0, -1.0, 1.0]], [-1.0])])
    sel = dense(node("vconcat", [grow, far, eq]),
                [[1, -1, 1], [-1, 0, -1], [0, 1, 0]], [0, 1, 0])
    pv = node("vslice", [Dd], start=7, stop=8)
    mem_next = node("vconcat", [pv, far], nid="mem_next")
    bl = np.zeros(8); bl[6] = 1.0
    dense(mem_next, np.zeros((8, 2)), bl, nid="logits")

    M9 = node("slice", ["x0"], start=9, stop=10)
    m1 = node("gmax", [node("mul", [M9, Z])])
    k2 = clip_s(node("sub", [Z, m1]), [-1.0], -4.0)
    mask2 = node("mul", [M9, k2])
    m2 = node("gmax", [node("mul", [mask2, Z])])
    k3 = clip_s(node("sub", [Z, m2]), [-1.0], -4.0)
    mask3 = node("mul", [mask2, k3])

    e1 = node("sub", [M9, mask2])
    e2 = node("sub", [mask2, mask3])
    E = node("concat", [e1, e2, mask3])
    ES = node("mul", [E, sel])
    tb = conv1(node("mul", [Dpar, eq]), [[0.01]])
    Zt = node("add", [Z, tb])
    ESZ = node("mul", [ES, Zt])
    A12 = node("mul", [node("slice", ["x0"], start=12, stop=13), betw])
    CA = node("concat", [ESZ, A12])
    conv1(CA, [[1.0, 1.0, 1.0, 1000.0]], nid="click")

    return {"graph": G, "memory_size": 2, "params": P}

import numpy as np
from collections import deque

POS = [(25,24,180),(33,21,0),(38,32,270),(33,40,90),(25,45,0),(14,40,180),(17,32,90),(14,21,270)]
GRID = {0:(0,1),1:(0,2),2:(1,2),3:(2,2),4:(2,1),5:(2,0),6:(1,0),7:(0,0)}
INV = {v:k for k,v in GRID.items()}
BUCKET_CENTER = {0:(32,20),2:(50,39),4:(32,57),6:(13,39)}  # (x,y)
BUCKET_BOX = {0:(29,18,6,5),2:(48,36,5,6),4:(29,55,6,5),6:(11,36,5,6)}

def _rot(a, r):
    k = ((-r) % 360) // 90
    return np.rot90(a, k) if k else a

def _templates():
    h = np.zeros((9,14), bool); h[:,0]=True; h[:,13]=True; h[8,:]=True
    d = np.zeros((17,17), bool); d[0,6]=True
    for r in range(1,7):
        for c in (6-r,7-r,5+r,6+r): d[r,c]=True
    for r in range(7,11): d[r,r+5]=True; d[r,r+6]=True
    for r in range(11,16):
        s=r-11; d[r,14-s]=True; d[r,15-s]=True
    d[16,10]=True
    return h, d

def _regions():
    R = {}
    I, J = np.indices((10,10))
    m = np.zeros((10,10),bool); m[0:5,:]=True; R[('d',0)]=m
    m = np.zeros((10,10),bool); m[:,5:10]=True; R[('d',2)]=m
    m = np.zeros((10,10),bool); m[5:10,:]=True; R[('d',4)]=m
    m = np.zeros((10,10),bool); m[:,0:5]=True; R[('d',6)]=m
    R[('d',1)] = J >= I
    R[('d',3)] = J >= 9-I
    R[('d',5)] = J <= I
    R[('d',7)] = J <= 9-I
    m = np.zeros((10,10),bool); m[0:3,3:7]=True; R[('b',0)]=m
    m = np.zeros((10,10),bool); m[7:10,3:7]=True; R[('b',4)]=m
    m = np.zeros((10,10),bool); m[3:7,0:3]=True; R[('b',6)]=m
    m = np.zeros((10,10),bool); m[3:7,7:10]=True; R[('b',2)]=m
    return R

REG = _regions()
MASK = np.ones((10,10),bool)
for _i in range(10):
    MASK[_i,_i]=False; MASK[_i,9-_i]=False

def _sim(canvas, ops):
    c = canvas.copy()
    for op,col in ops:
        c[REG[op]] = col
    return c

def _ok(c, target):
    return np.array_equal(c[MASK], target[MASK])

def _plan(canvas, target, colors, allow_b):
    unc = MASK.copy()
    rev = []
    ops = [k for k in REG if k[0]=='d' or allow_b]
    for _ in range(60):
        um = unc & (canvas != target)
        if not um.any():
            break
        best = None
        for op in ops:
            cov = REG[op] & unc
            n = int(cov.sum())
            if n == 0: continue
            vals = np.unique(target[cov])
            if len(vals) != 1 or int(vals[0]) not in colors: continue
            bad = int((cov & (canvas != target)).sum())
            sc = (bad > 0, n)
            if best is None or sc > best[0]:
                best = (sc, op, int(vals[0]))
        if best is None:
            return None
        rev.append((best[1], best[2]))
        unc = unc & ~REG[best[1]]
    else:
        return None
    fwd = rev[::-1]
    if not _ok(_sim(canvas, fwd), target):
        return None
    changed = True
    while changed:
        changed = False
        for i in range(len(fwd)):
            t = fwd[:i]+fwd[i+1:]
            if _ok(_sim(canvas, t), target):
                fwd = t; changed = True; break
    return fwd

def _swatches(obs):
    res = []
    for y in range(0, 16):
        for x in range(18, 60):
            b = obs[y:y+5, x:x+5]
            if b.shape != (5,5): continue
            if (b[0,:]==4).all() and (b[4,:]==4).all() and (b[:,0]==4).all() and (b[:,4]==4).all():
                inn = b[1:4,1:4]
                if (inn==inn[0,0]).all() and inn[0,0] != 4:
                    res.append((x, y, int(inn[0,0])))
    return res

def _basket_pos(obs):
    h, d = _templates()
    best, bp = -1, 0
    for p,(x,y,r) in enumerate(POS):
        t = _rot(h if p % 2 == 0 else d, r)
        th, tw = t.shape
        if y+th > 64 or x+tw > 64: continue
        sub = obs[y:y+th, x:x+tw]
        sc = float((sub[t]==2).sum()) / t.sum()
        if sc > best: best, bp = sc, p
    return bp

def _move(cur, goal):
    prev = {cur: None}; q = deque([cur])
    while q:
        p = q.popleft()
        if p == goal: break
        r, c = GRID[p]
        for a,(nr,nc) in (("ACTION1",(max(0,r-1),c)),("ACTION2",(min(2,r+1),c)),
                          ("ACTION3",(r,max(0,c-1))),("ACTION4",(r,min(2,c+1)))):
            if (nr,nc)==(1,1): continue
            n = INV[(nr,nc)]
            if n not in prev:
                prev[n] = (p,a); q.append(n)
    if goal not in prev: return "ACTION1"
    n = goal; act = None
    while prev[n] is not None:
        p,a = prev[n]; act = a; n = p
    return act or "ACTION1"

def act(obs, available, memory):
    obs = np.asarray(obs)
    canvas = obs[34:44, 27:37].astype(int)
    target = obs[3:13, 3:13].astype(int)
    sw = _swatches(obs)
    colors = {c for _,_,c in sw}
    cur = 15
    for x,y,c in sw:
        if y+5 < 64 and (obs[y+5, x:x+5]==0).all():
            cur = c; break
    bp = _basket_pos(obs)
    if bp % 2 == 0:
        bx,by,bw,bh = BUCKET_BOX[bp]
        memory['bucket'] = int((obs[by:by+bh, bx:bx+bw]==2).sum()) >= 8
    if _ok(canvas, target):
        return ("ACTION1", None, None)
    plan = memory.get('plan'); pre = memory.get('pre')
    valid = False
    if plan:
        if np.array_equal(canvas, pre[0]):
            valid = True
        elif len(plan) > 1 and np.array_equal(canvas, pre[1]):
            plan = plan[1:]; pre = pre[1:]; valid = True
    if not valid:
        bk = memory.get('bucket', None)
        plan = _plan(canvas, target, colors, bk is True)
        if plan is None and bk is None:
            plan = _plan(canvas, target, colors, True)
        if plan is None:
            plan = _plan(canvas, target, colors, True)
        if not plan:
            memory['plan'] = None
            return ("ACTION5", None, None)
        pre = []; c = canvas.copy()
        for op,col in plan:
            pre.append(c.copy()); c[REG[op]] = col
    memory['plan'] = plan; memory['pre'] = pre
    (kind, p), col = plan[0]
    if col != cur:
        for x,y,c in sw:
            if c == col:
                return ("ACTION6", y+2, x+2)
    if bp != p:
        return (_move(bp, p), None, None)
    if kind == 'd':
        return ("ACTION5", None, None)
    if memory.get('bucket') is False:
        memory['plan'] = None
        alt = _plan(canvas, target, colors, False)
        if alt:
            memory['plan'] = alt
            pre = []; c = canvas.copy()
            for op,cl in alt:
                pre.append(c.copy()); c[REG[op]] = cl
            memory['pre'] = pre
            (k2,p2),c2 = alt[0]
            if c2 != cur:
                for x,y,c in sw:
                    if c == c2: return ("ACTION6", y+2, x+2)
            if bp != p2: return (_move(bp,p2), None, None)
            return ("ACTION5", None, None)
        return ("ACTION5", None, None)
    bx, by = BUCKET_CENTER[p]
    return ("ACTION6", by, bx)

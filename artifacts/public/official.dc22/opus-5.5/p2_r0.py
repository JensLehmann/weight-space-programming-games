import numpy as np
from collections import deque
WSET={1,2,6,7,8,9,11,12,13,14,15}
MOVES={'ACTION1':(0,-2),'ACTION2':(0,2),'ACTION3':(-2,0),'ACTION4':(2,0)}
def _ok(b):
    if b.shape!=(2,2) or not set(b.flatten().tolist())<=WSET: return False
    a,b1,c,d=int(b[0,0]),int(b[0,1]),int(b[1,0]),int(b[1,1])
    return not(a==d and b1==c and a!=b1 and {a,b1}!={6,7})
def _comps(p):
    H,W=p.shape;s=np.zeros((H,W),bool);out=[]
    for y in range(H):
        for x in range(W):
            c=int(p[y,x])
            if s[y,x] or c in(0,5): continue
            q=[(y,x)];s[y,x]=True;pts=[]
            while q:
                cy,cx=q.pop();pts.append((cy,cx))
                for ny,nx in((cy+1,cx),(cy-1,cx),(cy,cx+1),(cy,cx-1)):
                    if 0<=ny<H and 0<=nx<W and not s[ny,nx] and int(p[ny,nx])==c:
                        s[ny,nx]=True;q.append((ny,nx))
            if len(pts)>=4: pts.sort();out.append((c,pts[len(pts)//2]))
    return out
def act(obs,available,memory):
    obs=np.asarray(obs).astype(int)
    rows=[r for r in range(63) if not np.all(obs[r]==3)]
    if not rows: return('ACTION1',None,None)
    oy,bot=rows[0],rows[-1]+1;pxl=64
    for x in range(16,64):
        col=obs[oy:bot,x]
        if np.all((col==0)|(col==5)): pxl=x;break
    board=obs[oy:bot,:pxl];panel=obs[oy:bot,pxl:];H,W=board.shape
    pp=np.argwhere(board==14)
    if len(pp)==0: return('ACTION1',None,None)
    py,px=int(pp[:,0].min()),int(pp[:,1].min());pos=(px,py)
    gp=np.argwhere(board==11)
    goal=(int(gp[:,1].min()),int(gp[:,0].min())) if len(gp) else None
    t=memory.get('t',0)+1;memory['t']=t
    seen=memory.setdefault('seen',{});walk={}
    for y in range(py%2,H-1,2):
        for x in range(px%2,W-1,2):
            b=board[y:y+2,x:x+2];walk[(x,y)]=_ok(b)
            if (x,y)!=pos: seen[(x,y)]=set(b.flatten().tolist())
    sig=bytes(bytearray(1 if(walk[k] or k==pos) else 0 for k in sorted(walk)))
    h=hash((sig,panel.tobytes()))
    bl=memory.setdefault('bl',set());last=memory.get('last')
    if last and last[0]==h and last[1]==pos: bl.add(last)
    memory['last']=None
    iv=memory.setdefault('iv',set());iv.add(pos)
    par={pos:None};dist={pos:0};dq=deque([pos])
    while dq:
        cur=dq.popleft()
        for a,(dx,dy) in MOVES.items():
            nb=(cur[0]+dx,cur[1]+dy)
            if nb in par or not walk.get(nb,False) or (h,cur,a) in bl: continue
            par[nb]=(cur,a);dist[nb]=dist[cur]+1;dq.append(nb)
    gain=memory.setdefault('gain',{})
    for c in par: gain.setdefault(c,t)
    def fm(tg):
        a=None
        while par[tg] is not None: tg,a=par[tg]
        return a
    def mv(a):
        memory['last']=(h,pos,a);return(a,None,None)
    if goal and goal in par and goal!=pos: return mv(fm(goal))
    btns=_comps(panel);tr=memory.setdefault('tr',set())
    def ck(b):
        memory['lb']=b;return('ACTION6',oy+b[1][0],pxl+b[1][1])
    u=seen.get(pos,set())
    if (6 in u or 7 in u) and ('tp',sig) not in tr:
        for b in btns:
            if b[0]==6: tr.add(('tp',sig));return ck(b)
    best=None
    for c in par:
        if c==pos or c in iv: continue
        cs=seen.get(c,set())
        if (6 in cs and 7 in cs) or (15 in cs and 2 in cs):
            if best is None or dist[c]<best[0]: best=(dist[c],c)
    if best: return mv(fm(best[1]))
    co=memory.setdefault('co',{})
    if sig not in co: co[sig]=sorted(par,key=lambda c:(-gain.get(c,0),-dist[c]))
    tg=[c for c in co[sig] if c in par];lb=memory.get('lb')
    order=sorted(btns,key=lambda b:b==lb)
    un=[b for b in order if (sig,b) not in tr]
    if un and tg:
        if tg[0]!=pos: return mv(fm(tg[0]))
        tr.add((sig,un[0]));tr.add((sig,pos,un[0]));return ck(un[0])
    for g in tg[:4]+[pos]:
        for b in order:
            if (sig,g,b) not in tr:
                if g!=pos: return mv(fm(g))
                tr.add((sig,g,b));return ck(b)
    n=memory.get('n',0)+1;memory['n']=n
    cells=[c for c in par if c!=pos]
    if cells:
        w=memory.get('w')
        if w not in par or w==pos: w=cells[(n*7)%len(cells)];memory['w']=w
        return mv(fm(w))
    if btns: return ck(btns[n%len(btns)])
    return mv(['ACTION1','ACTION2','ACTION3','ACTION4'][n%4])

import numpy as np
from collections import deque
import random
BASE=[[[1,1,0],[0,1,1],[1,0,1]],[[0,1,0],[0,1,0],[1,1,1]],[[1,0,1],[1,0,1],[1,1,1]],
      [[0,1,1],[1,0,1],[0,1,0]],[[0,1,0],[1,1,0],[0,1,1]],[[1,1,1],[0,0,1],[1,0,1]]]
COLS=[12,9,14,8]
LOOK={}
for s in range(6):
    a=np.array(BASE[s])
    for r in range(4):
        m=tuple(np.rot90(a,(4-r)%4).flatten().tolist())
        if m not in LOOK: LOOK[m]=(s,r)
DIRS={'ACTION1':(0,-1),'ACTION2':(0,1),'ACTION3':(-1,0),'ACTION4':(1,0)}
def ident(p,bg):
    m=tuple(int(v!=bg) for v in p.flatten());s,r=LOOK.get(m,(0,0))
    col=[v for v in p.flatten() if v!=bg]
    return s,(COLS.index(col[0]) if col and col[0] in COLS else 0),r
def act(obs,available,memory):
    try: return _act(obs,available,memory)
    except Exception:
        return (random.choice([a for a in DIRS if a in available] or list(available)),None,None)
def _act(obs,available,memory):
    obs=np.asarray(obs);mem=memory.setdefault('cells',{})
    types={};push={};goals={};player=None;dyn=[]
    for i in range(12):
        for j in range(12):
            x0=4+5*i;y0=5*j;p=obs[y0:y0+5,x0:x0+5]
            if x0<14 and y0>=50: types[(i,j)]='W';continue
            if int((p==4).sum())>=16:
                types[(i,j)]='W'
                for e,d in ((p[:,0],(-1,0)),(p[:,4],(1,0)),(p[0,:],(0,-1)),(p[4,:],(0,1))):
                    if (e==1).all(): push[(i+d[0],j+d[1])]=d
                continue
            if (p==5).all(): types[(i,j)]=mem.get((i,j),'U');continue
            if (p[0:2]==12).all() and (p[2:5]==9).all():
                player=(i,j);t=mem.get((i,j),'.')
                if t in ('G','F','U','W'): t='.'
                types[(i,j)]=t;mem[(i,j)]=t
                ls=dict(memory.get('lastsnap',()))
                if (i,j) in ls: dyn.append(((i,j),ls[(i,j)]))
                continue
            ring=np.concatenate([p[0,:],p[4,:],p[1:4,0],p[1:4,4]]);cen=p[1:4,1:4]
            if (ring==5).all() and (cen!=5).any(): t='G';goals[(i,j)]=ident(cen,5)
            elif int((cen==11).sum())==8: t='F'
            elif (p==1).any(): t='R'
            elif ((p==14)|(p==8)|(p==9)|(p==12)).any(): t='C'
            elif (p==0).any(): t='S'
            else: t='.'
            if t in ('S','C','R'): dyn.append(((i,j),t));t='.'
            types[(i,j)]=t;mem[(i,j)]=t
    for k,v in memory.get('goals',{}).items():
        if k not in goals and types.get(k)=='U': goals[k]=v
    memory['goals']=dict(goals)
    for k in goals: types[k]='G'
    snapnow=tuple(sorted(dyn));memory['lastsnap']=snapnow
    hist=memory.setdefault('hist',[])
    if not hist or memory.get('lastp')!=player: hist.append(snapnow)
    else: hist[-1]=snapnow
    memory['lastp']=player
    if len(hist)>200: del hist[:len(hist)-200]
    P=None
    for q in range(1,21):
        if len(hist)>=2*q and all(hist[-k]==hist[-k-q] for k in range(1,q+1)): P=q;break
    snaps=[dict(hist[len(hist)-P+i]) for i in range(P)] if P else [dict(snapnow)]
    NP=len(snaps)
    keyp=np.array([[obs[55+2*a,3+2*b] for b in range(3)] for a in range(3)])
    ks,kc,kr=ident(keyp,5)
    row=obs[61];L=0;cur=0;c=13
    while c<64 and row[c] in (3,11): L+=1;cur+=int(row[c]==11);c+=1
    mx=max(L,1);pv=memory.get('prev')
    if pv and player and pv[1]!=player and pv[0]-cur in (1,2): memory['dec']=pv[0]-cur
    memory['prev']=(cur,player)
    if player is None: return (random.choice([a for a in DIRS if a in available]),None,None)
    fidx={k:n for n,k in enumerate([k for k,t in types.items() if t=='F'])}
    def isw(cell): return types.get(cell,'W') in ('W','G')
    def eff(cell,ph,s,c,r,mask,rem):
        t=snaps[ph].get(cell) or types.get(cell,'.');rf=False
        if t=='S': s=(s+1)%6
        elif t=='C': c=(c+1)%4
        elif t=='R': r=(r+1)%4
        elif t=='F' and cell in fidx and not (mask>>fidx[cell])&1: mask|=1<<fidx[cell];rem=mx;rf=True
        return s,c,r,mask,rem,rf
    def plan(dec,explore,cur0):
        start=(player,ks,kc,kr,0,NP-1);best={start:cur0};q=deque([(start,cur0,None)])
        while q:
            st,rem,fa=q.popleft()
            if best.get(st,-1)>rem: continue
            pos,s,c,r,mask,ph=st;ph2=(ph+1)%NP
            for a,(dx,dy) in DIRS.items():
                if a not in available: continue
                n=(pos[0]+dx,pos[1]+dy);t=types.get(n,'W')
                if t=='W': continue
                f=fa or a
                if t=='G':
                    if goals.get(n)==(s,c,r) and rem-dec>=0: return f
                    continue
                if explore and t=='U': return f
                s2,c2,r2,m2,rm,rf=eff(n,ph2,s,c,r,mask,rem)
                if not rf:
                    rm=rem-dec
                    if rm<0: continue
                fin=n
                if n in push:
                    d=push[n];g=None
                    for k in range(1,12):
                        if isw((n[0]+d[0]*k,n[1]+d[1]*k)): g=k-1;break
                    if g:
                        fin=(n[0]+d[0]*g,n[1]+d[1]*g)
                        s2,c2,r2,m2,rm,_=eff(fin,ph2,s2,c2,r2,m2,rm)
                ns=(fin,s2,c2,r2,m2,ph2)
                if best.get(ns,-1)<rm: best[ns]=rm;q.append((ns,rm,f))
                if len(best)>300000: return None
        return None
    a=None
    for d in ([memory['dec']] if 'dec' in memory else [2,1]):
        a=plan(d,False,cur)
        if a: break
    if not a: a=plan(0,False,10**6) or plan(0,True,10**6)
    if not a:
        opts=[x for x,(dx,dy) in DIRS.items() if x in available and types.get((player[0]+dx,player[1]+dy),'W') not in ('W','G')]
        a=random.choice(opts or [x for x in DIRS if x in available])
    return (a,None,None)

import numpy as np
from collections import deque
SH=[[[1,1,0],[0,1,1],[1,0,1]],[[0,1,0],[0,1,0],[1,1,1]],[[1,0,1],[1,0,1],[1,1,1]],
    [[0,1,1],[1,0,1],[0,1,0]],[[0,1,0],[1,1,0],[0,1,1]],[[1,1,1],[0,0,1],[1,0,1]]]
COLS=[12,9,14,8]
LOOK={}
for _s in range(6):
    for _k in range(4):
        LOOK.setdefault(tuple((np.rot90(np.array(SH[_s]),-_k)>0).flatten().tolist()),(_s,_k))
MOVES=[("ACTION1",0,-1),("ACTION2",0,1),("ACTION3",-1,0),("ACTION4",1,0)]
def parse_pix(pix):
    pix=np.array(pix);m=pix!=5
    if not m.any():return None
    col=int(pix[m][0])
    if col not in COLS:return None
    sk=LOOK.get(tuple(m.flatten().tolist()))
    if sk is None:return None
    return (sk[0],COLS.index(col),sk[1])
def act(obs,available,memory):
    obs=np.asarray(obs).astype(int)
    mp=memory.setdefault("map",{});grid={};player=None
    per=np.ones((5,5),bool);per[1:4,1:4]=False
    for j in range(12):
        for i in range(12):
            x0,y0=4+5*i,5*j;b=obs[y0:y0+5,x0:x0+5]
            if i<=1 and j>=10:grid[(i,j)]="W";continue
            if (b[:2]==12).all() and (b[2:]==9).all():
                player=(i,j);t=mp.get((i,j),"E")
                if t=="R":t="E"
                mp[(i,j)]=t;grid[(i,j)]=t;continue
            inner=b[1:4,1:4]
            if (b==4).sum()>=10:t="W"
            elif (b[per]==5).all() and (inner!=5).any():
                req=parse_pix(inner);t=("G",req) if req is not None else "W"
            elif (b==5).all():grid[(i,j)]=mp.get((i,j),"E");continue
            elif (b==11).sum()>=8:t="R"
            elif (inner==14).any() or (inner==8).any():t="COL"
            elif (inner==1).any() and (inner==0).any():t="ROT"
            elif (inner==0).sum()>=2:t="SHP"
            else:t="E"
            mp[(i,j)]=t;grid[(i,j)]=t
    push={}
    for y in range(60):
        for i in range(12):
            x0=4+5*i
            if (obs[y,x0:x0+5]==1).all():
                if y%5==4:push[(i,y//5+1)]=(0,1)
                elif y%5==0:push[(i,y//5-1)]=(0,-1)
    for x in range(4,64):
        for j in range(12):
            if (obs[5*j:5*j+5,x]==1).all():
                ci=(x-4)//5
                if (x-4)%5==0:push[(ci-1,j)]=(-1,0)
                elif (x-4)%5==4:push[(ci+1,j)]=(1,0)
    cur=0;mx=0;c=13
    while c<64 and obs[61,c] in (3,11):
        mx+=1
        if obs[61,c]==11:cur+=1
        c+=1
    if mx==0:mx=cur=42
    key=parse_pix([[obs[55+2*r,3+2*cc] for cc in range(3)] for r in range(3)]) or (0,0,0)
    d=memory.get("d",2);prev=memory.get("prev")
    if prev is not None and player is not None and prev[1]!=player:
        diff=prev[0]-cur
        if diff in (1,2):d=diff
    memory["d"]=d;memory["prev"]=(cur,player)
    if player is None:return ("ACTION1",None,None)
    refs=[p for p,t in grid.items() if t=="R"];ridx={p:k for k,p in enumerate(refs)}
    def wallish(p):
        t=grid.get(p,"W");return t=="W" or isinstance(t,tuple)
    def effect(p,k,mk,st,dec,dd):
        t=grid.get(p,"E");s,cl,r=k
        if t=="COL":cl=(cl+1)%4
        elif t=="ROT":r=(r+1)%4
        elif t=="SHP":s=(s+1)%6
        k=(s,cl,r)
        if t=="R" and not (mk>>ridx[p])&1:mk|=1<<ridx[p];st=mx
        elif dec:
            st-=dd
            if st<0:return None
        return k,mk,st
    def bfs(dd):
        best={(player,key,0):cur};q=deque([(player,key,0,cur,None)])
        while q:
            pp,k,mk,st,first=q.popleft()
            for a,di,dj in MOVES:
                p=(pp[0]+di,pp[1]+dj);t=grid.get(p,"W")
                if t=="W":continue
                if isinstance(t,tuple):
                    if t[1]==k:return first or a
                    continue
                res=effect(p,k,mk,st,True,dd)
                if res is None:continue
                nk,nm,ns=res
                if p in push:
                    pdx,pdy=push[p];kk=None
                    for n in range(1,12):
                        if wallish((p[0]+pdx*n,p[1]+pdy*n)):kk=n;break
                    if kk is not None and kk-1>0:
                        p=(p[0]+pdx*(kk-1),p[1]+pdy*(kk-1))
                        nk,nm,ns=effect(p,nk,nm,ns,False,dd)
                sk=(p,nk,nm)
                if best.get(sk,-1)>=ns:continue
                best[sk]=ns;q.append((p,nk,nm,ns,first or a))
        return None
    a=None
    for dd in (d,1,0):
        a=bfs(dd)
        if a:break
    if a and a in available:return (a,None,None)
    for a,di,dj in MOVES:
        t=grid.get((player[0]+di,player[1]+dj),"W")
        if t!="W" and not isinstance(t,tuple) and a in available:return (a,None,None)
    return (available[0],None,None)

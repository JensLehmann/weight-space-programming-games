import numpy as np

def _markers(o):
    res=[]
    for r in range(1,62):
        for c in range(1,63):
            b=o[r-1:r+2,c-1:c+2]
            if o[r,c]!=4 and (b==4).sum()==8:
                res.append((r,c,int(o[r,c])))
    return res

def _boxes(o):
    res=[]
    for r in range(0,60):
        for c in range(0,60):
            if o[r,c]==2 and o[r,c+1]==2 and o[r+1,c]==2 and o[r+1,c+1] not in (2,4,5,0) and (r==0 or o[r-1,c]!=2) and (c==0 or o[r,c-1]!=2):
                res.append((r+2,c+2,int(o[r+1,c+1])))
    return res

def _mv(memory,cy,cx,dy,dx):
    vert=abs(dy)>=abs(dx)
    last=memory['last']
    if last and last[0]==(cy,cx) and last[1]==vert:
        memory['blk']+=1
        if vert and dx!=0: vert=False
        elif (not vert) and dy!=0: vert=True
    else:
        memory['blk']=0
    memory['last']=((cy,cx),vert)
    if vert:
        return ("ACTION1" if dy<0 else "ACTION2",None,None)
    return ("ACTION3" if dx<0 else "ACTION4",None,None)

def _unc(o,r,c):
    return (o[r-1:r+2,c-1:c+2]==4).sum()==8

def act(obs, available, memory):
    o=np.array(obs)[:63]
    if 'mk' not in memory:
        memory['mk']=_markers(o); memory['bx']=_boxes(o)
        memory['done']=set(); memory['last']=None; memory['n5']=0
        memory['seen']={}; memory['tgt']=None; memory['blk']=0
    h=hash(o.tobytes())
    memory['seen'][h]=memory['seen'].get(h,0)+1
    def sw(c=None):
        if c is not None: memory['done'].add(c)
        memory['tgt']=None; memory['blk']=0
        memory['n5']+=1
        if memory['n5']>8:
            memory['done']=set(); memory['n5']=0
        return ("ACTION5",None,None)
    z=np.argwhere(o==0)
    if len(z)==0:
        return sw()
    cy,cx=map(int,z[0])
    nb=o[max(0,cy-2):cy+3,max(0,cx-2):cx+3].flatten()
    nb=[int(v) for v in nb if v not in (0,4,5)]
    if not nb:
        return sw()
    col=max(set(nb),key=nb.count)
    if memory['seen'][h]>=3 or memory['blk']>=4:
        memory['seen']={}
        return sw(col)
    if col in memory['done']:
        return sw()
    t=memory['tgt']
    if t and t[2]==col and (t[0],t[1])!=(cy,cx):
        memory['n5']=0
        return _mv(memory,cy,cx,t[0]-cy,t[1]-cx)
    memory['tgt']=None
    uns=[(r,c,k) for r,c,k in memory['mk'] if _unc(o,r,c)]
    mk=[(r,c) for r,c,k in uns if k==col]
    if mk:
        seen={(cy,cx)}; st=[(cy,cx)]
        while st:
            y,x=st.pop()
            for a in (-1,0,1):
                for b in (-1,0,1):
                    ny,nx=y+a,x+b
                    if 0<=ny<63 and 0<=nx<64 and (ny,nx) not in seen and o[ny,nx]==col:
                        seen.add((ny,nx)); st.append((ny,nx))
        best=(-1,0,0,0)
        for dy in range(-63,64,3):
            if not (0<=cy+dy<64): continue
            for dx in range(-63,64,3):
                if not (0<=cx+dx<64): continue
                s=0
                for r,c in mk:
                    if (r-dy,c-dx) in seen: s+=1
                d=abs(dx)+abs(dy)
                if s>best[0] or (s==best[0] and d<best[3]):
                    best=(s,dy,dx,d)
        s,dy,dx,d=best
        if d==0 or s<=0:
            return sw(col)
        memory['n5']=0
        memory['tgt']=(cy+dy,cx+dx,col)
        return _mv(memory,cy,cx,dy,dx)
    need=set(k for r,c,k in uns)
    bx=[(r,c) for r,c,k in memory['bx'] if k in need and k!=col]
    if bx:
        r,c=min(bx,key=lambda p:abs(p[0]-cy)+abs(p[1]-cx))
        dy=r-cy; dx=c-cx
        if abs(dy)<2 and abs(dx)<2:
            dy=dy or 3
        memory['n5']=0
        return _mv(memory,cy,cx,dy,dx)
    return sw(col)

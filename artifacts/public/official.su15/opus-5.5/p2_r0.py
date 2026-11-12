import numpy as np, math
LV={10:(0,1),6:(1,2),15:(2,3),11:(3,4),12:(4,5),8:(5,7),9:(6,8),7:(7,9),14:(8,10)}
EN={7:1,14:2,13:3}
def comps(mask):
    H,W=mask.shape;seen=np.zeros_like(mask);out=[]
    for y,x in zip(*np.nonzero(mask)):
        if seen[y,x]:continue
        st=[(y,x)];seen[y,x]=True;pts=[]
        while st:
            a,b=st.pop();pts.append((a,b))
            for da in(-1,0,1):
                for db in(-1,0,1):
                    na,nb=a+da,b+db
                    if 0<=na<H and 0<=nb<W and mask[na,nb] and not seen[na,nb]:
                        seen[na,nb]=True;st.append((na,nb))
        ys=[p[0] for p in pts];xs=[p[1] for p in pts]
        out.append((min(xs),min(ys),max(xs)-min(xs)+1,max(ys)-min(ys)+1))
    return out
def squares(obs,y0,y1,cl,border=None):
    res=[]
    for c,(l,s) in LV.items():
        ys,xs=np.nonzero(obs[y0:y1]==c)
        for y,x in sorted(zip(ys+y0,xs)):
            if cl[y,x] or x+s>64:continue
            ye=min(y+s,y1)
            if not np.all(obs[y:ye,x:x+s]==c) or np.any(cl[y:ye,x:x+s]):continue
            if border is not None:
                ok=True
                for yy in range(y-1,y+s+1):
                    for xx in range(x-1,x+s+1):
                        if(yy in(y-1,y+s) or xx in(x-1,x+s)) and 0<=yy<y1 and 0<=xx<64 and obs[yy,xx]!=border:ok=False
                if not ok:continue
            cl[y:ye,x:x+s]=True
            res.append(dict(l=l,s=s,x=x,y=y,cx=x+s//2,cy=y+s//2))
    return res
def cap(px,py,x,y,w,h):
    qx=min(max(px,x),x+w-1);qy=min(max(py,y),y+h-1)
    return (px-qx)**2+(py-qy)**2<=64
def clampc(px,py,s):
    return min(max(px-s//2,0),64-s),min(max(py-s//2,10),min(63,64-s))
def ov(ax,ay,aw,bx,by,bw):
    return ax<bx+bw and bx<ax+aw and ay<by+bw and by<ay+aw
def evalc(px,py,m,p,F,E,eok=False):
    if not(10<=py<=62 and 0<=px<=63):return None
    cd=[f for f in F if cap(px,py,f['x'],f['y'],f['s'],f['s'])]
    if not any(f is m for f in cd):return None
    for f in cd:
        if (f is not m and f is not p) or f['l']!=m['l']:return None
    nx,ny=clampc(px,py,m['s'])
    for f in F:
        if any(f is g for g in cd):continue
        if ov(nx,ny,m['s'],f['x'],f['y'],f['s']) and (f is not p or f['l']!=m['l']):return None
    pen=0
    if not eok:
        for e in E:
            if cap(px,py,e[0],e[1],5,4):pen+=5
            dd=math.hypot(e[0]+2-(nx+m['s']//2),e[1]+2-(ny+m['s']//2))
            if dd<m['s']/2+5:pen+=3
            elif dd<m['s']/2+10:pen+=1
    return pen,nx+m['s']//2,ny+m['s']//2
def drag(m,T,p,F,E,ban,eok=False):
    dx=T[0]-m['cx'];dy=T[1]-m['cy'];d=math.hypot(dx,dy)
    if d<0.5:return None
    a0=math.atan2(dy,dx);best=None
    for a in(0,25,-25,50,-50,80,-80):
        an=a0+math.radians(a)
        for st in range(min(int(math.ceil(d)),15),0,-1):
            px=int(round(m['cx']+st*math.cos(an)));py=int(round(m['cy']+st*math.sin(an)))
            if(px,py) in ban:continue
            r=evalc(px,py,m,p,F,E,eok)
            if r is None:continue
            pr=d-math.hypot(T[0]-r[1],T[1]-r[2])
            if pr<=0.3:continue
            sc=r[0]*4-pr
            if best is None or sc<best[0]:best=(sc,px,py)
    return best
def emove(e,T,tol,F,E,ban,partner=None):
    cx,cy=e[0]+2,e[1]+2;d0=math.hypot(cx-T[0],cy-T[1]);best=None
    for a in range(0,360,8):
        r=math.radians(a)
        px=int(round(cx+5*math.cos(r)));py=int(round(cy+5*math.sin(r)))
        if(px,py) in ban or not(10<=py<=62 and 0<=px<=63):continue
        dx,dy=px-cx,py-cy;d=math.hypot(dx,dy)
        if d<1e-6:continue
        ux,uy=dx/d,dy/d;x,y=e[0],e[1]
        for _ in range(4):
            x=min(max(int(round(x+ux*3.4)),0),59);y=min(max(int(round(y+uy*3.4)),10),60)
        ex,ey=x+2,y+2;dist=math.hypot(ex-T[0],ey-T[1])
        if max(abs(ex-T[0]),abs(ey-T[1]))<=tol:sc=0
        elif dist>=d0-0.5:continue
        else:sc=1+dist
        for f in F:
            if cap(px,py,f['x'],f['y'],f['s'],f['s']):sc+=30
        for o in E:
            if o is not e and o is not partner and cap(px,py,o[0],o[1],5,4):sc+=15
        if best is None or sc<best[0]:best=(sc,px,py)
    return best
def act(obs,available,memory):
    obs=np.asarray(obs)
    ban=memory.setdefault('ban',set());seen=memory.setdefault('seen',{})
    key=obs.tobytes();seen[key]=seen.get(key,0)+1
    if memory.get('prev')==key and memory.get('pa') is not None:ban.add(memory['pa'])
    memory['prev']=key;memory['pa']=None
    def click(px,py):
        memory['pa']=(px,py);memory['lc']=(px,py);return('ACTION6',int(py),int(px))
    if np.any(obs[0:10]==2):
        if memory.get('lc') is not None:ban.add(memory['lc']);memory['lc']=None
        if memory.get('lastU')==key or 'ACTION7' not in available:
            memory['lastU']=None;return('RESET',None,None)
        memory['lastU']=key;return('ACTION7',None,None)
    memory['lastU']=None
    cl=np.zeros((64,64),bool);F=squares(obs,10,63,cl);goals=[];E=[]
    for c in(9,2,0):
        for(x,y,w,h) in comps((obs[10:63]==c)&~cl[10:63]):
            if w>=7 and h>=7:goals.append((x,y+10,w,h))
    for c,el in EN.items():
        for(x,y,w,h) in comps((obs[10:63]==c)&~cl[10:63]):
            if w>=4 and h>=3:E.append((x,y+10,el))
    tc=np.zeros((64,64),bool);R=squares(obs,0,10,tc,border=4);req={}
    for r in R:req[r['l']]=req.get(r['l'],0)+1
    ereq={}
    for c,el in EN.items():
        for(x,y,w,h) in comps((obs[0:10]==c)&~tc[0:10]):
            if w==5 and h==4:ereq[el]=ereq.get(el,0)+1
    def ing(cx,cy):return any(g[0]<=cx<g[0]+g[2] and g[1]<=cy<g[1]+g[3] for g in goals)
    def freeg(ex):
        o=[(any(g[0]<=f['cx']<g[0]+g[2] and g[1]<=f['cy']<g[1]+g[3] for f in F if f is not ex),g) for g in goals]
        o.sort(key=lambda t:t[0]);return[g for _,g in o]
    stuck=seen[key]>4
    for L,n in(sorted(req.items()) if not stuck else[]):
        if not goals:break
        if len([f for f in F if f['l']==L and ing(f['cx'],f['cy'])])>=n:continue
        best=None
        for f in [f for f in F if f['l']==L and not ing(f['cx'],f['cy'])]:
            for g in freeg(f)[:max(1,n)]:
                r=drag(f,(g[0]+g[2]//2,g[1]+g[3]//2),None,F,E,ban)
                if r and(best is None or r[0]<best[0]):best=r
        if best:return click(best[1],best[2])
        pool=[f for f in F if f['l']<L]
        for k in sorted(set(f['l'] for f in pool)):
            fs=[f for f in pool if f['l']==k]
            if len(fs)<2:continue
            pairs=sorted((math.hypot(a['cx']-b['cx'],a['cy']-b['cy']),i,j) for i,a in enumerate(fs) for j,b in enumerate(fs) if i<j)
            for d,i,j in pairs[:6]:
                a,b=fs[i],fs[j];mx=(a['cx']+b['cx'])//2;my=(a['cy']+b['cy'])//2
                if(mx,my) not in ban:
                    r=evalc(mx,my,a,b,F,E)
                    if r is not None and cap(mx,my,b['x'],b['y'],b['s'],b['s']) and r[0]<5:return click(mx,my)
                bb=None
                for m,o in((a,b),(b,a)):
                    r=drag(m,(o['cx'],o['cy']),o,F,E,ban)
                    if r and(bb is None or r[0]<bb[0]):bb=r
                if bb:return click(bb[1],bb[2])
        hi=[f for f in F if f['l']>L]
        if hi and E:
            f=min(hi,key=lambda f:f['l']);e=min(E,key=lambda e:math.hypot(e[0]+2-f['cx'],e[1]+2-f['cy']))
            r=drag(f,(e[0]+2,e[1]+2),None,F,E,ban,True)
            if r:return click(r[1],r[2])
    for el,n in(sorted(ereq.items()) if not stuck else[]):
        if not goals:break
        if len([e for e in E if e[2]==el and ing(e[0]+2,e[1]+2)])>=n:continue
        best=None
        for e in [e for e in E if e[2]==el and not ing(e[0]+2,e[1]+2)]:
            for g in goals:
                if any(g[0]<=o[0]+2<g[0]+g[2] and g[1]<=o[1]+2<g[1]+g[3] for o in E if o is not e):continue
                r=emove(e,(g[0]+g[2]//2,g[1]+g[3]//2),3,F,E,ban)
                if r and(best is None or r[0]<best[0]):best=r
        if best:return click(best[1],best[2])
        low=[e for e in E if e[2]==el-1]
        if len(low)>=2:
            d,a,b=min((math.hypot(a[0]-b[0],a[1]-b[1]),a,b) for i,a in enumerate(low) for b in low[i+1:])
            mx=(a[0]+b[0])//2+2;my=(a[1]+b[1])//2+2
            if cap(mx,my,a[0],a[1],5,4) and cap(mx,my,b[0],b[1],5,4) and(mx,my) not in ban and 10<=my<=62 and not any(cap(mx,my,f['x'],f['y'],f['s'],f['s']) for f in F):
                return click(mx,my)
            bb=None
            for m,o in((a,b),(b,a)):
                r=emove(m,(o[0]+2,o[1]+2),3,F,E,ban,o)
                if r and(bb is None or r[0]<bb[0]):bb=r
            if bb:return click(bb[1],bb[2])
    rng=np.random.default_rng(len(ban)+int(obs.sum())%997+seen[key])
    if F:
        f=F[int(rng.integers(len(F)))]
        r=drag(f,(int(rng.integers(5,59)),int(rng.integers(14,59))),None,F,E,ban)
        if r:return click(r[1],r[2])
    for _ in range(50):
        px=int(rng.integers(0,64));py=int(rng.integers(10,63))
        if(px,py) not in ban:return click(px,py)
    return click(32,40)

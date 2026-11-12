import numpy as np
N=20;S=3;OFF=2
D4=((1,0),(-1,0),(0,1),(0,-1))

def grid(obs):
    return [[int(obs[OFF+S*y+1][OFF+S*x+1]) for x in range(N)] for y in range(N)]

def nb(p,cs):
    return any((p[0]+dx,p[1]+dy) in cs for dx,dy in D4)

def comps(cells):
    cells=set(cells);out=[]
    while cells:
        s=cells.pop();st=[s];c={s}
        while st:
            x,y=st.pop()
            for dx,dy in D4:
                n=(x+dx,y+dy)
                if n in cells:
                    cells.remove(n);c.add(n);st.append(n)
        out.append(c)
    return out

def mk(body,conn):
    allp=set(body)|set(conn)
    px=min(p[0] for p in allp);py=min(p[1] for p in allp)
    w=max(p[0] for p in allp)-px+1;h=max(p[1] for p in allp)-py+1
    return (px,py),{'body':frozenset((x-px,y-py) for x,y in body),
        'conn':[(x-px,y-py,i) for i,(x,y) in enumerate(sorted(conn))],'w':w,'h':h}

def rot(sh):
    h=sh['h']
    return {'body':frozenset((h-1-y,x) for x,y in sh['body']),
            'conn':[(h-1-y,x,c) for x,y,c in sh['conn']],'w':sh['h'],'h':sh['w']}

def clamp(pos,sh):
    return (max(0,min(pos[0],N-sh['w'])),max(0,min(pos[1],N-sh['h'])))

def absb(pos,sh):
    return {(pos[0]+a,pos[1]+b) for a,b in sh['body']}

def key(sh):
    return (sh['body'],tuple(sorted((a,b) for a,b,_ in sh['conn'])))

def selparse(g,bg,grey):
    body=set();e8=set();e3=set()
    for y in range(N):
        for x in range(N):
            v=g[y][x]
            if v==8: e8.add((x,y))
            elif v==3: e3.add((x,y))
            elif (grey and v not in (bg,4)) or ((not grey) and v==0): body.add((x,y))
    conn=set(e8) if grey else {p for p in e8 if nb(p,body)}
    conn|={p for p in e3 if nb(p,body)}
    return body,conn

def spr(pos,sh):
    return {'known':True,'pos':pos,'shape':sh,'kind':None,'tested':False}

def init(m,g,bg):
    allc={(x,y):g[y][x] for y in range(N) for x in range(N)}
    grey=bg!=4 and any(v==4 for v in allc.values())
    m['grey']=grey;sp=[];sel=None
    if grey:
        b,c=selparse(g,bg,True)
        if b:
            pos,sh=mk(b,c);sp.append(spr(pos,sh));sel=0
        for comp in comps([p for p,v in allc.items() if v==4]):
            sp.append({'known':False,'cells':comp,'kind':None,'tested':False})
    else:
        bodies=[]
        for col in set(allc.values())-{bg,8,3}:
            bodies+=[(col,cc) for cc in comps([p for p,v in allc.items() if v==col])]
        non0=[cc for col,cc in bodies if col!=0];zeros=[cc for col,cc in bodies if col==0]
        selb=None
        for z in zeros:
            for cc in non0:
                if any(nb(p,cc) for p in z):
                    cc|=z;break
            else:
                if selb is None or len(z)>len(selb): selb=z
        groups=non0+([selb] if selb else [])
        conns=[set() for _ in groups]
        for p,v in allc.items():
            if v in (8,3):
                hits=[i for i,cc in enumerate(groups) if nb(p,cc)]
                if v==8: hits=hits[:1]
                for i in hits: conns[i].add(p)
        for i,cc in enumerate(groups):
            pos,sh=mk(cc,conns[i]);sp.append(spr(pos,sh))
        if selb: sel=len(groups)-1
    m['sp']=sp;m['sel']=sel
    order=([sel] if sel is not None else [])+[i for i in range(len(sp)) if i!=sel]
    m['disc']=order;m['queue']=[];m['failed']=set();m['cur']=None;m['pend']=None

def handle(m,g,bg,pend):
    sp=m['sp'];grey=m['grey']
    if pend[0]=='reveal':
        s=sp[pend[1]]
        if grey or not s.get('known'):
            b,c=selparse(g,bg,grey)
            if b:
                pos,sh=mk(b,c);s.update({'known':True,'pos':pos,'shape':sh})
        return
    s=sp[pend[1]];ppos,psh=pend[2],pend[3]
    b,c=selparse(g,bg,grey)
    if not b:
        s['kind']='fixed';s['tested']=True;return
    opos,osh=mk(b,c)
    if s['kind'] is None:
        psr=rot(psh);pp=clamp(ppos,psr)
        if absb(pp,psr)==b:
            s['kind']='rot';s['pos'],s['shape']=pp,psr;s['tested']=True
        elif absb(ppos,psh)==b:
            s['kind']='fixed';s['tested']=True
        else:
            s['kind']='stack';s['seq']=[psh,osh];s['pos'],s['shape']=opos,osh
        return
    s['seq'].append(osh);s['pos'],s['shape']=opos,osh
    seq=s['seq'];ks=[key(x) for x in seq]
    turns=[i for i in range(1,len(ks)-1) if ks[i-1]==ks[i+1]]
    if len(turns)>=2:
        L=seq[turns[0]:turns[1]+1]
        var=[{'body':sh['body'],'conn':[(a,b,(vi,j)) for a,b,j in sh['conn']],'w':sh['w'],'h':sh['h']} for vi,sh in enumerate(L)]
        s['var']=var;s['k']=len(L)-2;s['d']=-1;s['shape']=var[s['k']];s['tested']=True
    elif len(seq)>16:
        s['kind']='fixed';s['tested']=True

def ppstep(k,d,n):
    if 0<=k+d<n: return k+d,d
    d=-d
    if 0<=k+d<n: return k+d,d
    return k,d

def opts(s):
    out=[];sh=s['shape'];pos=s['pos']
    if s['kind']=='rot':
        for k in range(4):
            out.append((sh,k,pos));sh=rot(sh);pos=clamp(pos,sh)
    elif s['kind']=='stack' and 'var' in s:
        var=s['var'];n=len(var);k,d=s['k'],s['d'];seen={k:(0,pos)}
        for st in range(1,2*n+2):
            k,d=ppstep(k,d,n);pos=clamp(pos,var[k])
            if k not in seen: seen[k]=(st,pos)
        for k,(st,p) in seen.items(): out.append((var[k],st,p))
    else:
        out.append((sh,0,pos))
    return out

def plan(m):
    sp=m['sp'];ids=[i for i,s in enumerate(sp) if s.get('known')]
    if not ids: return False
    O={i:opts(sp[i]) for i in ids}
    sols=[];seen=set();placed={};cnt={};nodes=[0]
    def put(i,sh,ox,oy):
        for a,b,cd in sh['conn']: cnt.setdefault((ox+a,oy+b),[]).append((i,cd))
    def unput(sh,ox,oy):
        for a,b,cd in sh['conn']:
            l=cnt[(ox+a,oy+b)];l.pop()
            if not l: del cnt[(ox+a,oy+b)]
    def rec():
        nodes[0]+=1
        if len(sols)>=300 or nodes[0]>150000: return
        un=None
        for c,l in cnt.items():
            if len(l)==1: un=c;break
        if un is None:
            if len(placed)==len(ids):
                k=tuple(sorted(placed.items()))
                if k not in seen:
                    seen.add(k)
                    sig=frozenset(frozenset(l) for l in cnt.values())
                    sols.append((dict(placed),sig))
            return
        for i in ids:
            if i in placed: continue
            for oi,(sh,pr,pp) in enumerate(O[i]):
                for cx,cy,cid in sh['conn']:
                    ox,oy=un[0]-cx,un[1]-cy
                    if any(len(cnt.get((ox+a,oy+b),()))>=2 for a,b,_ in sh['conn']): continue
                    placed[i]=(oi,ox,oy);put(i,sh,ox,oy);rec();unput(sh,ox,oy);del placed[i]
    a0=ids[0]
    for oi,(sh,pr,pp) in enumerate(O[a0]):
        placed[a0]=(oi,0,0);put(a0,sh,0,0);rec();unput(sh,0,0);del placed[a0]
    sel=m['sel'];best=None
    for sol,sig in sols:
        if sig in m['failed']: continue
        lox=max(-ox for oi,ox,oy in sol.values());loy=max(-oy for oi,ox,oy in sol.values())
        hix=min(N-O[i][oi][0]['w']-ox for i,(oi,ox,oy) in sol.items())
        hiy=min(N-O[i][oi][0]['h']-oy for i,(oi,ox,oy) in sol.items())
        for Tx in range(lox,hix+1):
            for Ty in range(loy,hiy+1):
                cost=0
                for i,(oi,ox,oy) in sol.items():
                    sh,pr,pp=O[i][oi]
                    c=pr+abs(ox+Tx-pp[0])+abs(oy+Ty-pp[1])
                    if c: cost+=c+(i!=sel)
                    if best is not None and cost>=best[0]: break
                if best is None or cost<best[0]: best=(cost,sol,sig,Tx,Ty)
    if best is None: return False
    cost,sol,sig,Tx,Ty=best;m['cur']=sig;q=[]
    order=sorted(sol.keys(),key=lambda i:i!=sel)
    for i in order:
        oi,ox,oy=sol[i];sh,pr,pp=O[i][oi];tx,ty=ox+Tx,oy+Ty
        if pr+abs(tx-pp[0])+abs(ty-pp[1])==0: continue
        if i!=sel or q: q.append(('click',i))
        q+=[('A5',i)]*pr
        dx=tx-pp[0];dy=ty-pp[1]
        q+=[('mv',1 if dx>0 else -1,0)]*abs(dx)+[('mv',0,1 if dy>0 else -1)]*abs(dy)
    if not q:
        m['failed'].add(sig);return False
    m['queue']=q;return True

def click(m,i):
    sp=m['sp'];occ={}
    def cells(s):
        if s.get('known'):
            b=absb(s['pos'],s['shape']);c={(s['pos'][0]+a,s['pos'][1]+bb) for a,bb,_ in s['shape']['conn']}
            return b,c
        return set(s['cells']),set()
    for s in sp:
        b,c=cells(s)
        for p in b|c: occ[p]=occ.get(p,0)+1
    b,c=cells(sp[i])
    cand=[p for p in b if occ.get(p)==1] or [p for p in c if occ.get(p)==1] or list(b) or list(c)
    p=sorted(cand)[len(cand)//2]
    m['sel']=i
    return ('ACTION6',OFF+S*p[1]+1,OFF+S*p[0]+1)

def act(obs, available, memory):
    m=memory;g=grid(obs);bg=int(obs[63][63])
    if 'sp' not in m: init(m,g,bg)
    if m.get('pend'):
        p=m['pend'];m['pend']=None;handle(m,g,bg,p)
    sp=m['sp']
    while m['disc']:
        i=m['disc'][0];s=sp[i]
        if s['tested'] or (not s.get('known') and m['sel']==i):
            m['disc'].pop(0);continue
        if m['sel']!=i:
            m['pend']=('reveal',i);return click(m,i)
        if not s.get('known'):
            m['disc'].pop(0);continue
        m['pend']=('test',i,s['pos'],s['shape'])
        return ('ACTION5',None,None)
    if not m['queue']:
        if m['cur'] is not None: m['failed'].add(m['cur']);m['cur']=None
        if not plan(m):
            m['failed']=set()
            if not plan(m): return ('ACTION5',None,None)
    it=m['queue'].pop(0)
    if it[0]=='click': return click(m,it[1])
    sel=m['sel']
    if it[0]=='A5':
        s=sp[sel] if sel is not None else None
        if s is not None and s.get('known'):
            if s['kind']=='rot':
                s['shape']=rot(s['shape']);s['pos']=clamp(s['pos'],s['shape'])
            elif s['kind']=='stack' and 'var' in s:
                s['k'],s['d']=ppstep(s['k'],s['d'],len(s['var']))
                s['shape']=s['var'][s['k']];s['pos']=clamp(s['pos'],s['shape'])
        return ('ACTION5',None,None)
    dx,dy=it[1],it[2]
    if sel is not None and sp[sel].get('known'):
        s=sp[sel];nx,ny=s['pos'][0]+dx,s['pos'][1]+dy
        if nx>=0 and ny>=0 and nx+s['shape']['w']<=N and ny+s['shape']['h']<=N: s['pos']=(nx,ny)
    a={(0,-1):'ACTION1',(0,1):'ACTION2',(-1,0):'ACTION3',(1,0):'ACTION4'}[(dx,dy)]
    return (a,None,None)

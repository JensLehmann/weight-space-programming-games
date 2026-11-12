import numpy as np
def act(obs, available, memory):
    o=np.asarray(obs);H,W=o.shape
    def bad(r,c,cols):
        r0,r1,c0,c1=max(0,r-2),min(H,r+3),max(1,c-2),min(W,c+3)
        return np.isin(o[r0:r1,c0:c1],cols).any()
    if not memory.get('plan'):
        legs=[];bodies={}
        for r in range(1,H-1):
            for c in range(2,W-1):
                v=o[r,c];nb=[o[r-1,c],o[r+1,c],o[r,c-1],o[r,c+1]]
                if v not in(0,3,5) and all(n in(0,3) for n in nb):legs.append((r,c,v))
                if v==6 and len(set(nb))==1 and nb[0] not in(0,3,5):bodies[int(nb[0])]=(r,c)
        plan=[]
        for col,(br,bc) in bodies.items():
            ls=[l for l in legs if l[2]==col]
            pts=[(r,c) for r in range(H) for c in range(1,W) if o[r,c]==col and (abs(r-br)>3 or abs(c-bc)>3)]
            pts=[p for p in pts if all(abs(p[0]-l[0])+abs(p[1]-l[1])>2 for l in ls)]
            if not pts or not ls:continue
            tr=int(round(np.mean([p[0] for p in pts])));tc=int(round(np.mean([p[1] for p in pts])))
            dr,dc=tr-br,tc-bc;n=len(ls);cr,cc=br,bc
            for (r,c,_) in ls:
                for f in (1.0,0.5):
                    nr=min(H-3,max(2,r+int(dr*f)));nc=min(W-3,max(3,c+int(dc*f)))
                    pr=cr+(nr-r)/n;pc=cc+(nc-c)/n
                    if not bad(nr,nc,[2,10]) and not bad(int(pr),int(pc),[10]):
                        plan+=[(r,c),(nr,nc)];cr,cc=pr,pc;break
        if not plan:
            i=memory.get('i',0);memory['i']=i+1
            if legs:
                l=legs[i%len(legs)];plan=[(l[0],l[1])]
            else:plan=[(32,32)]
        memory['plan']=plan
    r,c=memory['plan'].pop(0)
    return ('ACTION6',int(r),int(c))

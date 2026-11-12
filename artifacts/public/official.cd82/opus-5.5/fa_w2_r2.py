import numpy as np
from collections import deque

def build():
    G=[]; P={}
    def mk(L,i,op,ins,**kw):
        d={"id":i,"op":op,"in":ins}; d.update(kw); L.append(d); return i
    n=lambda *a,**k: mk(G,*a,**k)
    cells=[(i,j) for i in range(10) for j in range(10) if i!=j and i+j!=9]
    pts=[(3+i,3+j) for i,j in cells]
    pts+=[(24,31),(21,39),(38,46),(46,49),(53,31),(56,24),(38,17),(31,14)]
    pts+=[(18,31),(39,52),(59,31),(38,11)]
    pts+=[(20,32),(38,50),(57,31),(38,13)]
    NP=len(pts)
    Wpa=np.zeros((2*NP,3,1,1))
    for k,(r,c) in enumerate(pts):
        Wpa[2*k,:,0,0]=[63,0,-r]; Wpa[2*k+1,:,0,0]=[0,63,-c]
    Wpd=np.zeros((NP,2*NP,1,1))
    for k in range(NP): Wpd[k,2*k,0,0]=-2; Wpd[k,2*k+1,0,0]=-2
    P["W_pa"]=Wpa; P["W_pd"]=Wpd; P["b_pd"]=np.full(NP,1.5)
    def reg(r,i,j):
        return [i<5,j>=i,j>=5,j>=9-i,i>=5,j<=i,j<5,j<=9-i,
                i<3 and 3<=j<7, 3<=i<7 and j>=7, i>=7 and 3<=j<7, 3<=i<7 and j<3][r]
    WM=np.zeros((12,80,1,1))
    for r in range(12):
        for k,(i,j) in enumerate(cells):
            if reg(r,i,j): WM[r,k,0,0]=4096.0
    P["W_M"]=WM
    P["W_bm"]=np.array([[-63,0,9],[0,63,-17]],float).reshape(2,3,1,1)
    n("cst","slice",["coords"],start=2,stop=3)
    n("pa","conv",["coords"],w="W_pa"); n("pab","abs",["pa"])
    n("pd","conv",["pab"],w="W_pd",b="b_pd"); n("pr0","relu",["pd"])
    n("PT","min2",["pr0","cst"])
    n("D80","slice",["PT"],start=0,stop=80); n("Bp","slice",["PT"],start=80,stop=88)
    n("Sp","slice",["PT"],start=88,stop=92); n("Cp","slice",["PT"],start=92,stop=96)
    n("M","conv",["D80"],w="W_M")
    n("bmr","conv",["coords"],w="W_bm"); n("bmrr","relu",["bmr"])
    n("bm0","slice",["bmrr"],start=0,stop=1); n("bm1","slice",["bmrr"],start=1,stop=2)
    n("bmn","min2",["bm0","bm1"]); n("bm","min2",["bmn","cst"])
    n("x2","slice",["x0"],start=2,stop=3)
    n("bbm","mul",["Bp","x2"]); n("bb","gmax",["bbm"])
    n("sbm","mul",["Sp","x2"]); n("sbv","gmax",["sbm"])
    n("done","vslice",["mem"],start=0,stop=12)
    n("cdone","vslice",["mem"],start=12,stop=13)
    n("latch","vslice",["mem"],start=13,stop=14)
    P["W_one4"]=np.ones((1,4)); n("sb","dense",["sbv"],w="W_one4")
    P["W_lq"]=np.array([[-1.,-1.]]); P["b_one1"]=np.ones(1); P["W_neg1"]=-np.ones((1,1))
    n("lqi","vconcat",["latch","sb"]); n("lq","dense",["lqi"],w="W_lq",b="b_one1")
    n("qn","vrelu",["lq"])
    n("latch_n","dense",["qn"],w="W_neg1",b="b_one1")
    n("nlat","dense",["latch"],w="W_neg1"); n("newly","vadd",["latch_n","nlat"])
    # constants for loop
    Ex=np.zeros((192,12))
    for r in range(12): Ex[r*16:(r+1)*16,r]=1
    P["W_ex"]=Ex; P["W_nex"]=-Ex; P["W_sum12"]=Ex.T.copy()
    P["b_1_192"]=np.ones(192); P["b_m1_192"]=-np.ones(192); P["b_1_12"]=np.ones(12)
    P["W_negI"]=-np.eye(12); P["W_I12"]=np.eye(12)
    Wg=np.zeros((12,25)); Wg[:,0:12]=-np.eye(12); Wg[:,12:24]=-np.eye(12); Wg[8:12,24]=-1
    P["W_gate"]=Wg; P["W_cov"]=np.full((1,12,1,1),1/4096.0)
    B=[]; b=lambda *a,**k: mk(B,*a,**k)
    b("Y","mul",["x0","W"])
    gl=[]
    for r in range(12):
        b("Mr%d"%r,"slice",["M"],start=r,stop=r+1)
        b("Yr%d"%r,"mul",["Y","Mr%d"%r]); gl.append(b("gr%d"%r,"gmean",["Yr%d"%r]))
    b("cnt","vconcat",gl)
    b("tot","dense",["cnt"],w="W_sum12")
    b("nte","dense",["tot"],w="W_nex",b="b_1_192")
    b("eqa","vadd",["cnt","nte"]); b("eq","vrelu",["eqa"])
    b("qa","dense",["tot"],w="W_negI",b="b_1_12"); b("q","vrelu",["qa"])
    b("gin","vconcat",["q","peeled","qn"])
    b("gta","dense",["gin"],w="W_gate",b="b_1_12"); b("gate","vrelu",["gta"])
    b("gE","dense",["gate"],w="W_ex",b="b_m1_192")
    b("vla","vadd",["eq","gE"]); b("valid","vrelu",["vla"])
    b("npk","dense",["valid"],w="W_sum12")
    b("peeled_n","vadd",["peeled","npk"])
    b("colsel_n","vadd",["colsel","valid"])
    b("tinc","dense",["peeled_n"],w="W_negI",b="b_1_12")
    b("tp_n","vadd",["tp","tinc"])
    b("cvm","mul",["M","npk"]); b("covp","conv",["cvm"],w="W_cov")
    b("Wd","sub",["W","covp"]); b("W_n","relu",["Wd"])
    G.append({"id":"H","op":"loop","T":13,
              "state":[{"id":"W","init":"cst"},{"id":"peeled","zeros":[12]},
                       {"id":"colsel","zeros":[192]},{"id":"tp","zeros":[12]}],
              "body":B,"next":{"W":"W_n","peeled":"peeled_n","colsel":"colsel_n","tp":"tp_n"}})
    n("negd","dense",["done"],w="W_negI"); n("ea","vadd",["H.peeled","negd"]); n("e","vrelu",["ea"])
    n("tp1","dense",["H.tp"],w="W_I12",b="b_1_12"); n("es","vmul",["e","tp1"])
    P["b_eps"]=0.01*np.arange(12); n("s","dense",["es"],w="W_I12",b="b_eps")
    Wp=np.zeros((144,12))
    for r in range(12):
        for q in range(12):
            if r!=q: Wp[r*12+q,r]+=200; Wp[r*12+q,q]-=200
    P["W_pd200"]=Wp; P["b_1_144"]=np.ones(144)
    n("a1l","dense",["s"],w="W_pd200",b="b_1_144"); n("a1","vrelu",["a1l"])
    n("a2l","dense",["s"],w="W_pd200"); n("a2","vrelu",["a2l"])
    Wc=np.zeros((12,288))
    for r in range(12): Wc[r,r*12:(r+1)*12]=1; Wc[r,144+r*12:144+(r+1)*12]=-1
    P["W_wc"]=Wc; P["b_m11"]=np.full(12,-11.0)
    n("a12","vconcat",["a1","a2"]); n("wcl","dense",["a12"],w="W_wc",b="b_m11"); n("win","vrelu",["wcl"])
    n("ce","dense",["win"],w="W_ex"); n("csw","vmul",["H.colsel","ce"])
    S16=np.zeros((16,192))
    for r in range(12): S16[:,r*16:(r+1)*16]=np.eye(16)
    P["W_s16"]=S16; n("cvec","dense",["csw"],w="W_s16")
    Wpm=np.zeros((8,12))
    for r in range(8): Wpm[r,r]=1
    for m in range(4): Wpm[[0,2,4,6][m],8+m]=1
    P["W_pm"]=Wpm; n("pp","dense",["win"],w="W_pm")
    Wis=np.zeros((1,12)); Wis[0,8:]=1; P["W_iss"]=Wis; n("isS","dense",["win"],w="W_iss")
    # move table
    gp={0:(0,1),1:(0,2),2:(1,2),3:(2,2),4:(2,1),5:(2,0),6:(1,0),7:(0,0)}
    inv={v:k for k,v in gp.items()}
    dlt={1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
    def nxt(s,a):
        r,c=gp[s]; r=min(2,max(0,r+dlt[a][0])); c=min(2,max(0,c+dlt[a][1]))
        return s if (r,c)==(1,1) else inv[(r,c)]
    dist=np.zeros((8,8),int)
    for t in range(8):
        dd={t:0}; dq=deque([t])
        while dq:
            u=dq.popleft()
            for v in range(8):
                if v not in dd and any(nxt(v,a)==u for a in dlt): dd[v]=dd[u]+1; dq.append(v)
        for v in range(8): dist[v,t]=dd[v]
    Wmv=np.zeros((4,64)); Wat=np.zeros((1,64)); Web=np.zeros((64,8)); Wep=np.zeros((64,8))
    for s0 in range(8):
        for t in range(8):
            Web[s0*8+t,s0]=1; Wep[s0*8+t,t]=1
            if s0==t: Wat[0,s0*8+t]=1; continue
            for a in (1,2,3,4):
                if dist[nxt(s0,a),t]==dist[s0,t]-1: Wmv[a-1,s0*8+t]=1; break
    P["W_mv"]=Wmv; P["W_at"]=Wat; P["W_eb"]=Web; P["W_ep"]=Wep
    n("pb","dense",["bb"],w="W_eb"); n("ppe","dense",["pp"],w="W_ep"); n("pair","vmul",["pb","ppe"])
    n("mvoh","dense",["pair"],w="W_mv"); n("atp","dense",["pair"],w="W_at")
    n("ph0","dense",["cdone"],w="W_neg1",b="b_one1")
    P["W_mvg"]=np.array([[1.,-1.]]); n("mvi","vconcat",["cdone","atp"])
    n("mvlin","dense",["mvi"],w="W_mvg"); n("mv","vrelu",["mvlin"])
    n("ca","vconcat",["cdone","atp","isS"])
    P["W_pr"]=np.array([[1.,1.,-1.]]); P["b_pr"]=np.array([-1.])
    P["W_st"]=np.array([[1.,1.,1.]]); P["b_st"]=np.array([-2.])
    n("prl","dense",["ca"],w="W_pr",b="b_pr"); n("prr","vrelu",["prl"])
    n("stl","dense",["ca"],w="W_st",b="b_st"); n("st","vrelu",["stl"])
    Wml=np.zeros((4,5)); Wml[:,:4]=np.eye(4); Wml[:,4]=1
    P["W_mvl"]=Wml; P["b_m1_4"]=-np.ones(4)
    n("mli","vconcat",["mvoh","mv"]); n("mll","dense",["mli"],w="W_mvl",b="b_m1_4"); n("mvl","vrelu",["mll"])
    Wlo=np.zeros((8,7))
    for a in range(4): Wlo[1+a,a]=1
    Wlo[5,4]=1; Wlo[6,5]=1; Wlo[6,6]=1
    P["W_lo"]=Wlo; P["b_lo"]=np.array([0,0,0,0,0,0.001,0,0])
    n("loi","vconcat",["mvl","prr","ph0","st"]); n("logits","dense",["loi"],w="W_lo",b="b_lo")
    P["W_one2"]=np.ones((1,2)); n("exi","vconcat",["prr","st"]); n("exs","dense",["exi"],w="W_one2")
    n("ex","vmul",["win","exs"])
    P["W_m1col"]=-np.ones((12,1)); n("nwe","dense",["newly"],w="W_m1col")
    n("dn1","vadd",["done","ex"]); n("dn2","vadd",["dn1","nwe"]); n("done_n","vrelu",["dn2"])
    n("cdone_n","vadd",["ph0","mv"])
    n("mem_next","vconcat",["done_n","cdone_n","latch_n"])
    n("cw","vmul",["cvec","ph0"]); n("xb","mul",["x0","bm"]); n("xc","mul",["xb","cw"])
    P["W_c16"]=np.ones((1,16,1,1)); n("cl1","conv",["xc"],w="W_c16")
    n("ws","vslice",["win"],start=8,stop=12); n("sv","vmul",["ws","st"])
    n("spm","mul",["Cp","sv"]); P["W_c4"]=np.ones((1,4,1,1)); n("cl2","conv",["spm"],w="W_c4")
    n("click","add",["cl1","cl2"])
    P={k:np.asarray(v,dtype=np.float32) for k,v in P.items()}
    return {"graph":G,"memory_size":14,"params":P}

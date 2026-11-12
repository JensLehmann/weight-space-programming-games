import numpy as np

def build():
    P = {}
    good = [2,6,7,8,9,10,11,12,13,14,15]
    taps = [(1,1),(1,2),(2,1),(2,2)]
    Wcnt = np.zeros((4,16,3,3))
    for (a,b) in taps:
        for c in good: Wcnt[0,c,a,b] = 20
        Wcnt[1,9,a,b] = 20; Wcnt[2,15,a,b] = 20; Wcnt[3,4,a,b] = 20
    P["Wcnt"] = Wcnt; P["bcnt"] = np.full(4,-10.0)
    Wtl = np.zeros((1,16,3,3)); Wtl[0,14,1,1]=1; Wtl[0,14,1,0]=-1; Wtl[0,14,0,1]=-1
    P["Wtl"] = Wtl
    g = 0.98; st = [(0,2),(4,2),(2,0),(2,4)]
    Wsh = np.zeros((4,1,5,5))
    for d,(a,b) in enumerate(st): Wsh[d,0,a,b] = g
    P["Wsh"] = Wsh
    Wsh2 = np.zeros((8,2,5,5))
    for ci in range(2):
        for d,(a,b) in enumerate(st): Wsh2[ci*4+d,ci,a,b] = g
    P["Wsh2"] = Wsh2
    Wl = np.zeros((8,12)); bl = np.zeros(8)
    for d in range(4): Wl[1+d,d] = 10.0
    Wl[6,4:12] = 9.0
    bl[0]=bl[5]=bl[7] = -1000.0; bl[6] = 0.3
    P["Wl"]=Wl; P["bl"]=bl
    Ws = np.zeros((2,8)); Ws[0,0:4]=1; Ws[1,4:8]=1; P["Ws"]=Ws
    Wn = np.zeros((1,16,3,3)); Wn[0,0]=10; Wn[0,5]=10
    P["Wn"]=Wn; P["bn"]=np.array([-5.0])
    Wnb = np.ones((1,16,1,1))
    for c in (0,3,4,5): Wnb[0,c,0,0]=0
    P["Wnb"]=Wnb
    Wcol = np.zeros((1,3,1,1)); Wcol[0,1,0,0]=50.0
    P["Wcol"]=Wcol; P["bcol"]=np.array([-31.0])
    th = 2*np.pi*0.13
    R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    P["Wr"]=R; P["br"]=(np.eye(2)-R)@np.array([1.0,0.0])
    P["Wt"]=np.array([[-0.5,0.0]]); P["bt"]=np.array([1.0])
    P["Wneg"]=-np.ones((1,1,1,1)); P["b2"]=np.array([2.0])
    P["Wcs"]=np.full((1,2,1,1),10.0); P["Wcf"]=np.ones((1,2,1,1))
    def n(i,op,ins,**k):
        d={"id":i,"op":op,"in":ins}; d.update(k); return d
    graph = [
        n("cnt","conv",["x0"],w="Wcnt",b="bcnt"), n("sg","sigmoid",["cnt"]),
        n("Wb","slice",["sg"],start=0,stop=1), n("s9","slice",["sg"],start=1,stop=2),
        n("s15","slice",["sg"],start=2,stop=3), n("s4","slice",["sg"],start=3,stop=4),
        n("k9","mul",["s9","s4"]), n("k15","mul",["s15","s4"]),
        n("G","slice",["x0"],start=11,stop=12),
        n("tlc","conv",["x0"],w="Wtl"), n("tl","relu",["tlc"]),
        {"id":"L","op":"loop","T":128,"state":[{"id":"v","zeros":[1,64,64]}],
         "body":[n("sh","conv",["v"],w="Wsh"),
                 n("m1","slice",["sh"],start=0,stop=1), n("m2","slice",["sh"],start=1,stop=2),
                 n("m3","slice",["sh"],start=2,stop=3), n("m4","slice",["sh"],start=3,stop=4),
                 n("ma","max2",["m1","m2"]), n("mb","max2",["m3","m4"]),
                 n("mc","max2",["ma","mb"]), n("md","max2",["mc","G"]),
                 n("vn","mul",["md","Wb"])],
         "next":{"v":"vn"}},
        n("sV","conv",["L.v"],w="Wsh"), n("q4","mul",["sV","tl"]), n("q","gmax",["q4"]),
        n("kk","concat",["k9","k15"]), n("kv","mul",["kk","L.v"]),
        n("skv","conv",["kv"],w="Wsh2"), n("qk8","mul",["skv","tl"]), n("qk","gmax",["qk8"]),
        n("qq","vconcat",["q","qk"]), n("logits","dense",["qq"],w="Wl",b="bl"),
        n("s","dense",["qk"],w="Ws"),
        n("colc","conv",["coords"],w="Wcol",b="bcol"), n("colm","sigmoid",["colc"]),
        n("nc","conv",["x0"],w="Wn",b="bn"), n("near","sigmoid",["nc"]),
        n("e9","slice",["x0"],start=9,stop=10), n("e15","slice",["x0"],start=15,stop=16),
        n("e915","concat",["e9","e15"]), n("bm0","mul",["e915","near"]),
        n("bm","mul",["bm0","colm"]), n("bs","mul",["bm","s"]),
        n("bsc","conv",["bs"],w="Wcs"),
        n("nbx","conv",["x0"],w="Wnb"), n("nbm","mul",["nbx","colm"]),
        n("tv","dense",["mem"],w="Wt",b="bt"),
        n("row","slice",["coords"],start=0,stop=1), n("dr","sub",["row","tv"]),
        n("ad","abs",["dr"]), n("sc","conv",["ad"],w="Wneg",b="b2"),
        n("ex","mul",["sc","nbm"]), n("cc","concat",["bsc","ex"]),
        n("click","conv",["cc"],w="Wcf"),
        n("mem_next","dense",["mem"],w="Wr",b="br"),
    ]
    params = {k: np.asarray(v,np.float32) for k,v in P.items()}
    return {"graph":graph,"memory_size":2,"params":params}

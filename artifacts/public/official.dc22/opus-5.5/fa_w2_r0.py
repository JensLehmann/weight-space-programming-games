import numpy as np
def build():
    P={};W=[1,2,6,7,8,9,11,12,13,14,15]
    Wf=np.zeros((3,16,3,3))
    for a,b in[(1,1),(1,2),(2,1),(2,2)]:
        for c in W:Wf[0,c,a,b]=1
        Wf[1,11,a,b]=1;Wf[2,14,a,b]=1
    P["Wf"]=Wf;P["bf"]=np.full(3,-3.)
    P["Wpc"]=np.ones((1,1,1,1));P["bpc"]=np.array([-10.])
    P["Wneg"]=-np.ones((1,1,1,1));P["bone"]=np.array([1.]);P["btwo"]=np.array([2.])
    Ws=np.zeros((4,1,5,5))
    for k,(a,b) in enumerate([(0,2),(4,2),(2,0),(2,4)]):Ws[k,0,a,b]=.97
    P["Ws"]=Ws;Wo=np.zeros((8,4))
    for k in range(4):Wo[1+k,k]=1000
    P["Wo"]=Wo;P["bo"]=np.array([-1.,0,0,0,0,-1,.01,-1])
    Wb=np.zeros((1,16,1,1))
    for c in range(16):
        if c not in(0,3,4,5):Wb[0,c,0,0]=1
    P["Wb"]=Wb;Wa=np.zeros((1,8));Wa[0,6]=.21;P["Wa"]=Wa
    P["Wk"]=np.array([[1000.]]);P["bk"]=np.array([-995.]);P["Wn1"]=np.array([[-1.]])
    def n(i,o,inp,**k):return dict(id=i,op=o,**({"in":inp} if inp is not None else {}),**k)
    def s(i,x,a):return n(i,"slice",[x],start=a,stop=a+1)
    body=[n("sh","conv",["v"],w="Ws"),s("s0","sh",0),s("s1","sh",1),s("s2","sh",2),s("s3","sh",3),
      n("m01","max2",["s0","s1"]),n("m23","max2",["s2","s3"]),n("ma","max2",["m01","m23"]),
      n("mw","mul",["ma","walk"]),n("vn","max2",["mw","goal"])]
    g=[s("x5","x0",5),n("cs","colsum",["x5"]),n("pz","conv",["cs"],w="Wpc",b="bpc"),n("pr","relu",["pz"]),
      s("one","coords",2),n("panel","min2",["pr","one"]),n("notP","conv",["panel"],w="Wneg",b="bone"),
      n("f","conv",["x0"],w="Wf",b="bf"),n("fr","relu",["f"]),s("w0","fr",0),s("g0","fr",1),s("p0","fr",2),
      n("walk","mul",["w0","notP"]),n("goal","mul",["g0","notP"]),n("pl","mul",["p0","notP"]),
      dict(id="H",op="loop",T=180,state=[{"id":"v","init":"goal"}],body=body,next={"v":"vn"}),
      n("D","conv",["H.v"],w="Ws"),n("Dp","mul",["D","pl"]),n("dv","gmax",["Dp"]),
      n("logits","dense",["dv"],w="Wo",b="bo"),n("bc","conv",["x0"],w="Wb"),n("btn","mul",["bc","panel"]),
      s("rr","coords",0),n("dd","sub",["rr","mem"]),n("ad","abs",["dd"]),
      n("sc","conv",["ad"],w="Wneg",b="btwo"),n("click","mul",["btn","sc"]),
      n("u0","dense",["aprev"],w="Wa"),n("u","vadd",["mem","u0"]),n("kz","dense",["u"],w="Wk",b="bk"),
      n("sg","vsigmoid",["kz"]),n("sn","dense",["sg"],w="Wn1"),n("mem_next","vadd",["u","sn"])]
    return {"graph":g,"memory_size":1,"params":{k:np.asarray(v,np.float32) for k,v in P.items()}}

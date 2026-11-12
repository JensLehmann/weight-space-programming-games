import numpy as np

def build():
    f=np.float32
    G=[]; P={}
    def cv(i,o,w,b):
        n="w_"+o; P[n]=np.array(w,f).reshape(1,-1,1,1); P["b_"+o]=np.array([b],f)
        G.append({"id":o,"op":"conv","in":[i],"w":n,"b":"b_"+o})
    def op(o,t,ins,**k):
        d={"id":o,"op":t,"in":ins}; d.update(k); G.append(d)
    op("p","slice",["x0"],start=9,stop=10)
    op("e","slice",["x0"],start=14,stop=15)
    op("r","slice",["coords"],start=0,stop=1)
    op("c","slice",["coords"],start=1,stop=2)
    op("pr","mul",["p","r"]); op("rp","gmax",["pr"])
    op("pcr","mul",["p","c"]); op("gr","gmax",["pcr"])
    cv("c","q",[-1.0],1.0)
    op("pq","mul",["p","q"]); op("gl","gmax",["pq"])
    op("gg","vconcat",["gr","gl"])
    P["Wmid"]=np.array([[0.5,-0.5]],f); P["bmid"]=np.array([0.5],f)
    op("mid","dense",["gg"],w="Wmid",b="bmid")
    op("d","sub",["rp","r"])
    cv("d","d2",[1.0],-7.5/63)
    op("ad2","abs",["d2"])
    cv("ad2","bl",[-21.0],1.0); op("band","relu",["bl"])
    op("dc","sub",["c","mid"]); op("adc","abs",["dc"])
    cv("adc","cml",[-21.0],1.0); op("colm","relu",["cml"])
    cv("adc","pxl",[-1.5],1.0); op("prox","relu",["pxl"])
    cv("dc","ril",[200.0],-200*4/63); op("ri","sigmoid",["ril"])
    cv("dc","lil",[-200.0],-200*4/63); op("li","sigmoid",["lil"])
    op("o3","slice",["x0"],start=7,stop=15)
    w=[0]*8; w[0]=1; w[3]=1; w[7]=1
    cv("o3","opn",w,0.0)
    op("ob","mul",["opn","band"]); op("obp","mul",["ob","prox"])
    op("rr","mul",["obp","ri"]); op("R","gmean",["rr"])
    op("ll","mul",["obp","li"]); op("L","gmean",["ll"])
    op("eb","mul",["e","band"]); op("ebc","mul",["eb","colm"]); op("C","gmax",["ebc"])
    op("v","vconcat",["L","R","C","aprev"])
    Wo=np.zeros((8,11),f); Wo[3,0]=1000; Wo[4,1]=1000; Wo[6,2]=100
    Wo[3,3+3]=1.5; Wo[4,3+4]=1.5
    P["Wo"]=Wo; P["bo"]=np.array([-10,0,0,0,0.1,0,-3,-10],f)
    op("logits","dense",["v"],w="Wo",b="bo")
    op("ec","mul",["e","colm"])
    op("cc","concat",["ebc","ec","e"])
    cv("cc","click",[10.0,2.0,1.0],0.0)
    return {"graph":G,"memory_size":0,"params":P}

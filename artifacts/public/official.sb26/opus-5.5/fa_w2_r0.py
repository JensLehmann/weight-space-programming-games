import numpy as np

def build():
    P = {}; G = []
    def n(i, op, ins, **kw):
        d = {"id": i, "op": op, "in": ins}; d.update(kw); G.append(d)
    Wcm = np.zeros((7, 3, 1, 1))
    Wcm[0,0]=126;Wcm[0,2]=-1;Wcm[1,0]=-126;Wcm[1,2]=3;Wcm[2,0]=126;Wcm[2,2]=-15
    Wcm[3,0]=-126;Wcm[3,2]=105;Wcm[4,0]=126;Wcm[4,2]=-107;Wcm[5,1]=1;Wcm[6,1]=-1;Wcm[6,2]=1
    P["Wcm"]=Wcm
    n("cm","conv",["coords"],w="Wcm"); n("cmr","relu",["cm"])
    n("ones","slice",["coords"],start=2,stop=3)
    for i in range(5): n("q%d"%i,"slice",["cmr"],start=i,stop=i+1)
    n("colf","slice",["cm"],start=5,stop=6); n("colm","slice",["cm"],start=6,stop=7)
    n("mr1a","min2",["q0","q1"]); n("mrow1","min2",["mr1a","ones"])
    n("mmida","min2",["q2","q3"]); n("mmid","min2",["mmida","ones"])
    n("mpal","min2",["q4","ones"])
    P["Wid"]=np.ones((1,1,1,1));P["bm1"]=np.array([-1.0]);P["b1"]=np.array([1.0])
    P["Wneg"]=-np.ones((1,1,1,1));P["b100"]=np.array([100.0]);P["Wsum"]=np.ones((1,16,1,1))
    Wsr=np.zeros((1,1,3,3));Wsr[0,0,1,0]=1;P["Wsr"]=Wsr
    n("blk","slice",["x0"],start=0,stop=1); n("redp","slice",["x0"],start=2,stop=3)
    n("Bm","mul",["blk","mmid"]); n("Rb","rowmax",["Bm"])
    Wu=np.zeros((2,1,7,7));Wu[0,0,0,3]=1;Wu[1,0,6,3]=1;P["Wu"]=Wu
    n("u","conv",["Rb"],w="Wu"); n("u0","slice",["u"],start=0,stop=1); n("u1","slice",["u"],start=1,stop=2)
    Wv=np.zeros((1,1,7,7));Wv[0,0,0,3]=1;P["Wv"]=Wv; n("v","conv",["u0"],w="Wv")
    Wup=np.zeros((1,1,3,3));Wup[0,0,0,1]=1;P["Wup"]=Wup; n("vv","conv",["v"],w="Wup")
    n("Mp","add",["v","u1"]); n("Mq","sub",["Mp","vv"]); n("Mr","conv",["Mq"],w="Wid",b="bm1"); n("M","relu",["Mr"])
    Wed=np.zeros((1,16,3,3));Wed[0,4,1,0]=3;Wed[0,4,1,1]=-3;Wed[0,4,1,2]=-1;Wed[0,4,2,2]=-1;P["Wed"]=Wed
    n("edp","conv",["x0"],w="Wed",b="bm1"); n("edr","relu",["edp"]); n("ed","min2",["edr","ones"])
    n("E","mul",["ed","M"])
    WH=np.zeros((1,16,3,3));WH[0,4,0,2]=1;P["WH"]=WH; n("H","conv",["x0"],w="WH"); n("EH","mul",["E","H"])
    Wr5=np.zeros((16,16,5,5))
    for c in range(16): Wr5[c,c,2,:]=1
    P["Wr5"]=Wr5;P["br5"]=-4.0*np.ones(16)
    n("r5p","conv",["x0"],w="Wr5",b="br5"); n("r5r","relu",["r5p"]); n("R5","mul",["r5r","mmid"])
    n("r5m","gmean",["R5"]); P["WL"]=np.eye(16)*(4096.0/12.0); n("L","dense",["r5m"],w="WL")
    n("xl0","mul",["x0","L"]); n("XL","conv",["xl0"],w="Wsum"); n("xlm","conv",["XL"],w="Wid",b="bm1")
    n("tw","mul",["EH","xlm"]); n("wgt","add",["E","tw"])
    G.append({"id":"PL","op":"loop","T":63,"state":[{"id":"ps","zeros":[1,64,64]}],
      "body":[{"id":"psw","op":"add","in":["ps","wgt"]},{"id":"psn","op":"conv","in":["psw"],"w":"Wsr"}],
      "next":{"ps":"psn"}})
    n("Dm","mul",["redp","E"]); n("pinv","conv",["PL.ps"],w="Wneg",b="b100"); n("Sm","mul",["Dm","pinv"])
    n("pp1","conv",["PL.ps"],w="Wid",b="b1"); n("cmk","mul",["EH","pp1"])
    n("xc","mul",["x0","cmk"]); n("Rcp1","gmax",["xc"])
    n("r5c","mul",["R5","colm"]); n("LCv","gmax",["r5c"])
    Wtr=np.zeros((1,16,3,3));Wtr[0,2,1,1]=1;Wtr[0,2,1,2]=1;Wtr[0,2,1,0]=-1;Wtr[0,2,0,1]=-1;P["Wtr"]=Wtr
    n("trp","conv",["x0"],w="Wtr",b="bm1"); n("trr","relu",["trp"]); n("tr","mul",["trr","mmid"])
    Ws3=np.zeros((16,16,7,7));Ws1=np.zeros((16,16,3,3));Wdr=np.zeros((16,16,3,3))
    for c in range(16): Ws3[c,c,0,3]=1;Ws1[c,c,0,1]=1;Wdr[c,c,0,0]=1
    P["Ws3"]=Ws3;P["Ws1"]=Ws1;P["Wdr"]=Wdr
    n("xs3","conv",["x0"],w="Ws3"); n("xs4","conv",["xs3"],w="Ws1")
    n("xlc","mul",["xs4","LCv"]); n("LCs","conv",["xlc"],w="Wsum"); n("jj","add",["colf","LCs"])
    P["Wj"]=np.array([[[[63.0/6.0]]]]);P["bj"]=np.array([-65.0/6.0]); n("jv","conv",["jj"],w="Wj",b="bj")
    # first pass: ranks of sub-frame calls
    n("xr1","mul",["xs4","Rcp1"]); n("Rs1","conv",["xr1"],w="Wsum")
    n("rk01","add",["Rs1","jv"]); n("val1","min2",["Rs1","ones"]); n("rv1","mul",["rk01","val1"])
    Wsc=np.zeros((16,16,7,7));bsc=-4.0*np.ones(16)
    for c in range(16):
        if c in (0,2,4): bsc[c]=-20.0; continue
        Wsc[c,c,3,3]=1;Wsc[c,c,3,4]=1;Wsc[c,c,4,3]=1;Wsc[c,4,4,4]=1;Wsc[c,c,4,6]=1
    P["Wsc"]=Wsc;P["bsc"]=bsc
    n("hsp","conv",["x0"],w="Wsc",b="bsc"); n("hsq","relu",["hsp"]); n("hsm","mul",["hsq","mmid"])
    n("hsr","conv",["hsm"],w="Wdr"); n("hrk","mul",["hsr","rv1"]); n("Rc2","gmax",["hrk"])
    n("Rcp","vadd",["Rcp1","Rc2"])
    n("xr","mul",["xs4","Rcp"]); n("Rsel","conv",["xr"],w="Wsum")
    n("rk0","add",["Rsel","jv"]); n("rk","conv",["rk0"],w="Wid",b="bm1")
    n("rinv","conv",["rk"],w="Wneg",b="b100"); n("val","min2",["Rsel","ones"])
    n("trv","mul",["tr","val"]); n("Ss","mul",["trv","rinv"])
    n("score","add",["Sm","Ss"]); n("mr","gmax",["score"])
    P["Wn1"]=np.array([[-1.0]]);P["b101"]=np.array([101.0]); n("kp1","dense",["mr"],w="Wn1",b="b101")
    Wtl=np.zeros((1,16,3,3))
    for c in range(16):
        if c in (4,5): Wtl[0,c,1,0]=1
        else: Wtl[0,c,1,1]=1
    P["Wtl"]=Wtl
    n("tlp","conv",["x0"],w="Wtl",b="bm1"); n("tl0","relu",["tlp"]); n("TL","mul",["tl0","mrow1"])
    G.append({"id":"TLL","op":"loop","T":63,"state":[{"id":"s","init":"TL"}],
      "body":[{"id":"sc","op":"conv","in":["s"],"w":"Wsr"},{"id":"sn","op":"add","in":["sc","TL"]}],
      "next":{"s":"sn"}})
    n("dd","sub",["TLL.s","kp1"]); n("da","abs",["dd"]); n("eqp","conv",["da"],w="Wneg",b="b1")
    n("eq","relu",["eqp"]); n("Ek","mul",["TL","eq"]); n("xe","mul",["x0","Ek"]); n("tc","gmax",["xe"])
    n("xt","mul",["x0","tc"]); n("Dt","conv",["xt"],w="Wsum"); n("D","mul",["Dt","mpal"])
    Whol=np.zeros((16,16,3,3));bhol=-3.0*np.ones(16)
    for c in range(16):
        if c in (0,2,4): bhol[c]=-10.0; continue
        Whol[c,c,1,1]=1;Whol[c,c,1,2]=1;Whol[c,c,2,1]=1;Whol[c,4,2,2]=1
    P["Whol"]=Whol;P["bhol"]=bhol
    n("hdp","conv",["x0"],w="Whol",b="bhol"); n("hdr","relu",["hdp"]); n("Hdet","mul",["hdr","mpal"])
    n("hv","gmax",["Hdet"]); n("hcm","gmean",["Hdet"]); n("Hd1","conv",["Hdet"],w="Wsum")
    W7=np.zeros((1,1,7,7));W7[0,0,0:4,0:4]=1;P["W7"]=W7
    n("HP0","conv",["Hd1"],w="W7"); n("HP","min2",["HP0","ones"]); n("nHP","conv",["HP"],w="Wneg",b="b1")
    n("ge0","sub",["TLL.s","kp1"]); n("ge1","conv",["ge0"],w="Wid",b="b1"); n("ge2","relu",["ge1"])
    n("ge","min2",["ge2","ones"]); n("nm0","mul",["TL","ge"]); n("nm","mul",["nm0","Dt"])
    n("needg","gmean",["nm"]); n("Pcg","gmean",["D"]); n("hct","vmul",["hcm","tc"])
    n("shc","vconcat",["needg","Pcg","hct"])
    Wsh=np.zeros((1,18));Wsh[0,0]=4096.0;Wsh[0,1]=-256.0;Wsh[0,2:]=3072.0;P["Wsh"]=Wsh
    n("shv","dense",["shc"],w="Wsh"); n("shr","vrelu",["shv"]); n("sh1","dense",["shr"],w="Wn1",b="b1")
    n("sh2","vrelu",["sh1"]); n("sh","dense",["sh2"],w="Wn1",b="b1")
    n("bs0","sub",["score","mr"]); n("bs1","conv",["bs0"],w="Wid",b="b1"); n("best","relu",["bs1"])
    n("xb","mul",["xs4","best"]); n("fc","gmax",["xb"])
    P["Wni16"]=-np.eye(16);P["b116"]=np.ones(16)
    n("nfc","dense",["fc"],w="Wni16",b="b116"); n("ts","vadd",["tc","sh"])
    n("t1","dense",["ts"],w="Wni16",b="b116"); n("t2","vrelu",["t1"]); n("tm","dense",["t2"],w="Wni16",b="b116")
    n("w1","vmul",["hv","nfc"]); n("want","vmul",["w1","tm"])
    P["Wo16"]=5.0*np.ones((1,16)); n("wsc","dense",["want"],w="Wo16"); n("uc","vtanh",["wsc"])
    n("nu","dense",["uc"],w="Wn1",b="b1")
    n("xw","mul",["x0","want"]); n("Dc0","conv",["xw"],w="Wsum"); n("Dc1","mul",["Dc0","mpal"])
    n("Dc","mul",["Dc1","HP"]); n("Dl","mul",["D","nHP"]); n("Dla","mul",["Dl","nu"])
    n("Dca","mul",["Dc","uc"]); n("Dfin","add",["Dla","Dca"])
    n("Bp","mul",["blk","mpal"]); P["W33"]=np.ones((1,1,3,3))
    n("Bd0","conv",["Bp"],w="W33"); n("Bd","min2",["Bd0","ones"]); n("DB","mul",["Dfin","Bd"])
    n("sel","gmax",["DB"]); n("nsel","dense",["sel"],w="Wn1",b="b1")
    n("ca","mul",["score","sel"]); n("cb","mul",["Dfin","nsel"]); n("click","add",["ca","cb"])
    Wlg=np.zeros((8,1));Wlg[6,0]=1;blg=np.zeros(8);blg[5]=0.5;P["Wlg"]=Wlg;P["blg"]=blg
    n("lg0","dense",["mr"],w="Wlg",b="blg")
    Wlm=np.zeros((8,1));Wlm[7,0]=-1000.0;blm=np.zeros(8);blm[7]=5500.0;P["Wlm"]=Wlm;P["blm"]=blm
    n("lg1","dense",["mem"],w="Wlm",b="blm"); n("logits","vadd",["lg0","lg1"])
    P["Wm1"]=np.array([[1.0]]); n("mem_next","dense",["mem"],w="Wm1",b="b1")
    params={k:np.asarray(v,dtype=np.float32) for k,v in P.items()}
    return {"graph":G,"memory_size":1,"params":params}

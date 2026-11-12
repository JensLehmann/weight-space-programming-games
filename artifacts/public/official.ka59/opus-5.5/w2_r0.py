import numpy as np

def build():
    p = {}
    W1 = np.zeros((32,19,5,5),np.float32); b1 = np.zeros(32,np.float32)
    # ch0/1: colour-0 pixels with a pixel below (excludes step bar) -> row/63, col/63
    for ch,coord in ((0,16),(1,17)):
        W1[ch,0,2,2]=1; W1[ch,coord,2,2]=1
        for c in range(16): W1[ch,c,3,2]+=1
        b1[ch]=-2
    # ch2/3: free top-left frame corners (S=4 only at such corners)
    K=2.0
    for ch in (2,3):
        W1[ch,4,2,2]+=K; W1[ch,4,2,3]+=K; W1[ch,4,3,2]+=K
        W1[ch,4,1,2]-=K; W1[ch,4,2,1]-=K; W1[ch,1,3,3]+=K
        W1[ch,16,2,2]=-1.0
        W1[ch,18,2,2]=1.0
        b1[ch]=-4*K
    W1[3,17,2,2]=-1.0/64.0
    p["W1"]=W1; p["b1"]=b1
    W2=np.zeros((32,32,3,3),np.float32)
    for c in range(4): W2[c,c,1,1]=1
    p["W2"]=W2; p["b2"]=np.zeros(32,np.float32)
    W3=np.zeros((32,96),np.float32)
    for c in range(4): W3[c,c]=1
    p["W3"]=W3; p["b3"]=np.zeros(32,np.float32)
    p["Wh"]=np.zeros((16,16,3,3),np.float32); p["Uh"]=np.zeros((16,32),np.float32); p["bh"]=np.zeros(16,np.float32)
    Wz=np.zeros((128,96),np.float32); bz=np.zeros(128,np.float32)
    s=1/63.0
    # g32=br, g33=bc, g34=1-r*, g35=1-r*-c*/4032 ; tr=1-g34+2s ; tc=64(g34-g35)+2s
    Wz[0,34]=-1; Wz[0,32]=-1; bz[0]=1+0.5*s          # down
    Wz[1,32]=1;  Wz[1,34]=1;  bz[1]=-1-3.5*s         # up
    Wz[2,34]=64; Wz[2,35]=-64; Wz[2,33]=-1; bz[2]=0.5*s   # right
    Wz[3,34]=-64; Wz[3,35]=64; Wz[3,33]=1; bz[3]=-3.5*s   # left
    p["Wz"]=Wz; p["bz"]=bz
    p["Wm"]=np.zeros((64,64),np.float32); p["Wmz"]=np.zeros((64,128),np.float32)
    p["Wma"]=np.zeros((64,8),np.float32); p["bm"]=np.zeros(64,np.float32)
    Wo=np.zeros((8,192),np.float32); bo=np.zeros(8,np.float32)
    Wo[1,1]=100; Wo[2,0]=100; Wo[3,3]=100; Wo[4,2]=100
    bo[0]=-10; bo[5]=-10; bo[6]=-10; bo[7]=-10
    p["Wo"]=Wo; p["bo"]=bo
    p["Wc"]=np.zeros((1,48),np.float32); p["bc"]=np.zeros(1,np.float32)
    return p

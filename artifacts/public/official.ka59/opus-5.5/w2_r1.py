import numpy as np

def build():
    S={"W1":(32,19,5,5),"b1":(32,),"W2":(32,32,3,3),"b2":(32,),"W3":(32,96),"b3":(32,),
       "Wh":(16,16,3,3),"Uh":(16,32),"bh":(16,),"Wz":(128,96),"bz":(128,),"Wm":(64,64),
       "Wmz":(64,128),"Wma":(64,8),"bm":(64,),"Wo":(8,192),"bo":(8,),"Wc":(1,48),"bc":(1,)}
    p={k:np.zeros(s) for k,s in S.items()}
    W1=p["W1"];b1=p["b1"]
    # ch0/1: selected centre (colour 0), bottom row suppressed
    W1[0,0,2,2]=1;W1[0,16,2,2]=-1;W1[0,18,3,2]=1;b1[0]=-1
    W1[1,0,2,2]=1;W1[1,17,2,2]=-1;W1[1,18,3,2]=1;b1[1]=-1
    # ch2/3: target outline colour 4, not near colour 14, bottom bar suppressed
    W1[2,4,2,2]=1;W1[2,16,2,2]=-1;W1[2,18,4,2]=1;b1[2]=-1
    W1[3,4,2,2]=1;W1[3,17,2,2]=-1;W1[3,18,4,2]=1;b1[3]=-1
    W1[2,14,:,:]-=1.0;W1[3,14,:,:]-=1.0
    # ch4: deselected centre (colour 4 with 14 above)
    W1[4,4,2,2]=1;W1[4,14,1,2]=1;b1[4]=-1
    # ch5: border of non-selected block
    W1[5,14,2,2]=1;W1[5,0,:,:]-=1.0
    # ch6: never-selected centre (colour 5)
    W1[6,5,2,2]=1
    for i in range(7):
        p["W2"][i,i,1,1]=1;p["W3"][i,i]=1
    Wz,bz=p["Wz"],p["bz"]
    defs=[(34,32,1),(34,32,-1),(35,33,1),(35,33,-1)]
    for k,(t,s,sg) in enumerate(defs):
        for j,off in enumerate((1.5,2.5)):
            u=2*k+j
            Wz[u,t]=63*sg;Wz[u,s]=-63*sg;bz[u]=-2*sg-off
    # position copies
    Wz[8,32]=1.0;Wz[9,33]=1.0
    Wm,Wmz,bm=p["Wm"],p["Wmz"],p["bm"]
    Wmz[0,8]=0.1;Wmz[1,9]=0.1
    K=1000.0
    for u,(zi,mi,sg) in zip(range(2,6),[(8,0,1),(8,0,-1),(9,1,1),(9,1,-1)]):
        Wmz[u,zi]=0.1*K*sg;Wm[u,mi]=-K*sg;bm[u]=-2.0
    Wo,bo=p["Wo"],p["bo"]
    for k in range(4):
        Wo[1+k,2*k]=1;Wo[1+k,2*k+1]=-1
    for u in range(2,6):
        Wo[6,128+u]=-1.0
    bo[0]=-10;bo[5]=-10;bo[7]=-10;bo[6]=-1.4
    p["Wc"][0,4]=1.0;p["Wc"][0,5]=0.1;p["Wc"][0,6]=1.0
    return {k:np.asarray(v,np.float32) for k,v in p.items()}

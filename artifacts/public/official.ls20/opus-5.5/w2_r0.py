import numpy as np

def build():
    P = {"W1": np.zeros((32,19,5,5)), "b1": np.zeros(32), "W2": np.zeros((32,32,3,3)), "b2": np.zeros(32),
         "W3": np.zeros((32,96)), "b3": np.zeros(32), "Wh": np.zeros((16,16,3,3)), "Uh": np.zeros((16,32)),
         "bh": np.zeros(16), "Wz": np.zeros((128,96)), "bz": np.zeros(128), "Wm": np.zeros((64,64)),
         "Wmz": np.zeros((64,128)), "Wma": np.zeros((64,8)), "bm": np.zeros(64), "Wo": np.zeros((8,192)),
         "bo": np.zeros(8), "Wc": np.zeros((1,48)), "bc": np.zeros(1)}
    W1, b1 = P["W1"], P["b1"]
    def k1(o, c, di, dj, v): W1[o, c, di+2, dj+2] += v
    for dj in range(-2, 3):
        for di in (0, 1): k1(0, 12, di, dj, 1.0)
        for di in (-1, 0): k1(1, 9, di, dj, 1.0)
    k1(0, 12, -1, 0, -10.0); b1[0] = -9.0
    k1(1, 9, 1, 0, -10.0); b1[1] = -9.0
    for di in (-1, 0, 1):
        for dj in (0, 1, 2):
            k1(2, 9, di, dj, 1.0); k1(3, 9, di, -dj, 1.0)
    k1(2, 9, 0, -1, -10.0); b1[2] = -8.0
    k1(3, 9, 0, 1, -10.0); b1[3] = -8.0
    k1(4, 4, 0, 0, 1.0)
    for c in (0, 1, 8, 14, 11): k1(5, c, 0, 0, 1.0)
    k1(14, 5, 0, 0, 1.0)
    def ramp(o, ch, s, b): k1(o, ch, 0, 0, s*126.0); b1[o] = b
    ramp(6,16,1,-105.0); ramp(7,16,1,-106.0); ramp(8,17,-1,23.0); ramp(9,17,-1,22.0)
    ramp(10,17,-1,7.0); ramp(11,17,-1,6.0); ramp(12,16,1,-119.0); ramp(13,16,1,-120.0)
    W2 = P["W2"]
    W2[0,0,2,1] = 1; W2[1,1,0,1] = 1; W2[2,2,1,2] = 1; W2[3,3,1,0] = 1
    W2[4,4,1,1] = 1; W2[5,5,1,1] = 1; W2[10,14,1,1] = 1
    for o, (a, b) in zip((6,7,8,9), ((6,7),(8,9),(10,11),(12,13))):
        W2[o,a,1,1] = 1; W2[o,b,1,1] = -1
    W3, b3 = P["W3"], P["b3"]
    for k in range(6): W3[k,k] = 1
    W3[9,10] = 1; W3[6,6] = 1; W3[6,7] = 1; b3[6] = -1.5; W3[7,8] = 1; W3[8,9] = 1
    Uh, bh, Wh = P["Uh"], P["bh"], P["Wh"]
    Uh[0,5] = 1e10; Uh[0,9] = 1e3
    for c in (4,6,7,8): Uh[0,c] = -1e14
    for (a, b) in ((0,1),(1,0),(1,2),(2,1)): Wh[0,0,a,b] = 0.25
    for k in range(1, 5):
        Uh[k,k-1] = 1e16; bh[k] = -1e16; Wh[k,0,1,1] = 1
    Wz, Wo, bo = P["Wz"], P["Wo"], P["bo"]
    for d in range(4): Wz[d,1+d] = 1; Wo[1+d,d] = 1
    bo[:] = -1.0; bo[1:5] = 0.0
    rev = {1:2, 2:1, 3:4, 4:3}
    for k in range(1, 5):
        P["Wma"][k,k] = 3.0; Wo[rev[k],128+k] = -1e13
    return {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}

import numpy as np

SHAPES = {"W1":(32,19,5,5),"b1":(32,),"W2":(32,32,3,3),"b2":(32,),"W3":(32,96),"b3":(32,),
"Wh":(16,16,3,3),"Uh":(16,32),"bh":(16,),"Wz":(128,96),"bz":(128,),"Wm":(64,64),"Wmz":(64,128),
"Wma":(64,8),"bm":(64,),"Wo":(8,192),"bo":(8,),"Wc":(1,48),"bc":(1,)}

def build():
    p = {k: np.zeros(s, np.float32) for k, s in SHAPES.items()}
    # ch0: run-button detector (full 5x5 colour-9 window) + down/right tie-break
    p["W1"][0, 9, :, :] = 1.0
    p["W1"][0, 16, 2, 2] = 0.3
    p["W1"][0, 17, 2, 2] = 0.3
    p["b1"][0] = -24.0
    # ch1: unset program cell (colour 1) -> strong click priority
    p["W1"][1, 1, 2, 2] = 10.0
    for c in (0, 1):
        p["W2"][c, c, 1, 1] = 1.0
        p["W3"][c, c] = 1.0
        p["Wc"][0, c] = 1.0
    p["bo"][6] = 1.0
    return p

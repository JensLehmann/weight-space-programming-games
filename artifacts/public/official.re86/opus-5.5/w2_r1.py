import numpy as np

def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32),
        "bh": (16,), "Wz": (128, 96), "bz": (128,), "Wm": (64, 64),
        "Wmz": (64, 128), "Wma": (64, 8), "bm": (64,), "Wo": (8, 192),
        "bo": (8,), "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    # memory unit k ~= 1 iff previous action was k
    for k in range(8):
        p["Wma"][k, k] = 4.0
    # fixed exploration cycle: up -> right -> down -> left -> switch piece -> up
    nxt = {1: 4, 4: 2, 2: 3, 3: 5, 5: 1, 0: 1, 6: 1, 7: 1}
    for k, n in nxt.items():
        p["Wo"][n, 128 + k] = 5.0
    p["bo"][1] = 0.5
    p["bo"][0] = -20.0
    p["bo"][6] = -20.0
    p["bo"][7] = -20.0
    return p

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
    acts = [2, 4, 1, 3]  # down, right, up, left
    nxt = {2: 4, 4: 2, 1: 3, 3: 1}
    for i, a in enumerate(acts):
        p["Wm"][i, i] = 0.9
        p["Wma"][i, a] = 0.15
        # hold current direction while counter small, hand off when large
        p["Wo"][a, 128 + i] = 4.0
        p["Wo"][nxt[a], 128 + i] = 9.0
    p["bo"][:] = -50.0
    p["bo"][2] = 1.0
    p["bo"][4] = 0.6
    p["bo"][1] = -1.0
    p["bo"][3] = -0.5
    return p

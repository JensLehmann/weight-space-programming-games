import numpy as np

SHAPES = {
    "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
    "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32),
    "bh": (16,), "Wz": (128, 96), "bz": (128,), "Wm": (64, 64),
    "Wmz": (64, 128), "Wma": (64, 8), "bm": (64,), "Wo": (8, 192),
    "bo": (8,), "Wc": (1, 48), "bc": (1,),
}

def build():
    p = {k: np.zeros(s, np.float32) for k, s in SHAPES.items()}
    for a in range(1, 5):
        p["Wma"][a, a] = 10.0
    p["bo"][:] = -10.0
    p["bo"][4] = 1.0                   # first move: right
    p["bo"][2] = 0.0
    p["Wo"][2, 128 + 4] = 15.0         # after right -> down
    p["Wo"][4, 128 + 4] = -5.0
    p["Wo"][4, 128 + 2] = 15.0         # after down -> right
    p["Wo"][2, 128 + 2] = -5.0
    p["Wo"][4, 128 + 1] = 15.0
    p["Wo"][4, 128 + 3] = 15.0
    return p

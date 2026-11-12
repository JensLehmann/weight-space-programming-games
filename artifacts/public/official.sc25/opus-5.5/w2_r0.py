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
    # colour-0 detector restricted to the lower area (spell grid)
    p["W1"][0, 0, 2, 2] = 10.0
    p["W1"][0, 16, 2, 2] = 10.0
    p["b1"][0] = -17.0
    p["W2"][0, 0, 1, 1] = 1.0
    p["W3"][0, 0] = 1.0
    # z0 = spatial max of f3 channel 0 (g index 32)
    p["Wz"][0, 32] = 1.0
    # policy: click while hint cells remain, else move left
    p["Wo"][6, 0] = 4.0
    p["bo"][3] = 1.0
    # click map follows the hint detector
    p["Wc"][0, 0] = 1.0
    return p

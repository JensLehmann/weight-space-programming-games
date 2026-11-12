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
    # memory units 0..7 copy previous action
    for k in range(8):
        p["Wma"][k, k] = 3.0
    p["bo"][:] = -3.0
    p["bo"][6] = 0.5   # first step: click
    # cycle: click -> left -> up -> right -> down -> left ...
    nxt = {6: 3, 3: 1, 1: 4, 4: 2, 2: 3}
    for prev, n in nxt.items():
        p["Wo"][n, 128 + prev] = 4.0
    p["Wo"][6, 128 + 6] = -6.0
    for prev in (1, 2, 3, 4):
        p["Wo"][6, 128 + prev] = -6.0
    # click map: peak at row 51, col 15 using coordinate features is not
    # available in f3/h (zero), so leave constant -> first cell clicked.
    return p

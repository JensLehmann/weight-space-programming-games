import numpy as np

def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,), "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8),
        "bm": (64,), "Wo": (8, 192), "bo": (8,), "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    for i in range(8):
        p["Wma"][i, i] = 4.0          # unit i: previous action
        p["Wm"][8 + i, i] = 4.0       # unit 8+i: action two steps ago
    succ = {4: 2, 2: 4}
    for a, n in succ.items():
        p["Wo"][a, 128 + a] += 4.0        # continue last action
        p["Wo"][a, 128 + 8 + a] += -8.0   # same twice -> stop continuing
        p["Wo"][n, 128 + 8 + a] += 6.0    # ... and turn
    p["bo"][:] = -30.0
    p["bo"][1:5] = 0.0
    p["bo"][4] = 0.5
    return p

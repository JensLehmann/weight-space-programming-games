import numpy as np

def build():
    shapes = {
        "W1": (32, 19, 5, 5), "b1": (32,), "W2": (32, 32, 3, 3), "b2": (32,),
        "W3": (32, 96), "b3": (32,), "Wh": (16, 16, 3, 3), "Uh": (16, 32), "bh": (16,),
        "Wz": (128, 96), "bz": (128,), "Wm": (64, 64), "Wmz": (64, 128), "Wma": (64, 8),
        "bm": (64,), "Wo": (8, 192), "bo": (8,), "Wc": (1, 48), "bc": (1,),
    }
    p = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
    # memory unit i ~ tanh(3 * [previous action == i])
    for i in range(8):
        p["Wma"][i, i] = 3.0
    # successor table: up, right, down, left, switch cross, repeat
    succ = {0: 1, 1: 4, 4: 2, 2: 3, 3: 5, 5: 1, 6: 1, 7: 1}
    for prev, nxt in succ.items():
        p["Wo"][nxt, 128 + prev] = 5.0
    # episode start -> ACTION1; never RESET / click / ACTION7
    p["bo"][:] = np.array([-10.0, 0.1, 0.0, 0.0, 0.0, 0.0, -10.0, -10.0], np.float32)
    return p

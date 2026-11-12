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
    # memory units 0..7: one-hot copy of the previous action
    for j in range(8):
        p["Wma"][j, j] = 3.0
    # unit 8: trace "left was pressed two steps ago" (copies unit 3 from previous memory)
    p["Wm"][8, 3] = 3.0
    p["bm"][8] = -1.5
    # cycle: up(1) -> left(3) -> left(3) -> down(2) -> select(5) -> up(1)
    p["Wo"][3, 128 + 1] = 10.0           # after up: left
    p["Wo"][3, 128 + 3] = 10.0           # after first left: left
    p["Wo"][2, 128 + 8] = 25.0           # after second left (trace set): down
    p["Wo"][3, 128 + 8] = -25.0
    p["Wo"][5, 128 + 2] = 10.0           # after down: select next cross
    p["Wo"][1, 128 + 5] = 10.0           # after select: up
    p["Wo"][1, 128 + 4] = 10.0
    p["bo"] = np.array([-50.0, 1.0, 0.0, 0.0, 0.0, 0.0, -50.0, -50.0], np.float32)
    return p

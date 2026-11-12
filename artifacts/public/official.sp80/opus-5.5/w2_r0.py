import numpy as np

def build():
    p = {
        "W1": np.zeros((32, 19, 5, 5), np.float32), "b1": np.zeros(32, np.float32),
        "W2": np.zeros((32, 32, 3, 3), np.float32), "b2": np.zeros(32, np.float32),
        "W3": np.zeros((32, 96), np.float32), "b3": np.zeros(32, np.float32),
        "Wh": np.zeros((16, 16, 3, 3), np.float32), "Uh": np.zeros((16, 32), np.float32),
        "bh": np.zeros(16, np.float32),
        "Wz": np.zeros((128, 96), np.float32), "bz": np.zeros(128, np.float32),
        "Wm": np.zeros((64, 64), np.float32), "Wmz": np.zeros((64, 128), np.float32),
        "Wma": np.zeros((64, 8), np.float32), "bm": np.zeros(64, np.float32),
        "Wo": np.zeros((8, 192), np.float32), "bo": np.zeros(8, np.float32),
        "Wc": np.zeros((1, 48), np.float32), "bc": np.zeros(1, np.float32),
    }
    ROW, COL = 16, 17
    SEL, CUP = 9, 11
    specs = [
        (SEL, COL, +1, -1.0),  # 0: right edge of selected platform
        (SEL, COL, -1, 0.0),   # 1: 1 - left edge
        (SEL, ROW, +1, -1.0),  # 2: bottom edge
        (SEL, ROW, -1, 0.0),   # 3: 1 - top edge
        (CUP, COL, +1, -1.0),  # 4: cups right edge
        (CUP, COL, -1, 0.0),   # 5: 1 - cups left edge
        (CUP, ROW, +1, -1.0),  # 6: cups bottom edge
        (CUP, ROW, -1, 0.0),   # 7: 1 - cups top edge
    ]
    for c, (colour, coord, sgn, bias) in enumerate(specs):
        p["W1"][c, colour, 2, 2] = 1.0
        p["W1"][c, coord, 2, 2] = sgn
        p["b1"][c] = bias
        p["W2"][c, c, 1, 1] = 1.0
        p["W3"][c, c] = 1.0

    # indices of f3.max entries inside g = [hmax(16), hmean(16), f3max(32), f3mean(32)]
    A, B, E, F, C, D, G, H = [32 + i for i in range(8)]
    Wz, bz = p["Wz"], p["bz"]

    def add(d1, d2, s2=1.0):
        out = dict(d1)
        for k, v in d2.items():
            out[k] = out.get(k, 0.0) + s2 * v
        return out

    neg = lambda d: {k: -v for k, v in d.items()}
    vx = {A: 63.0, B: -63.0, C: -63.0, D: 63.0}   # 2*(platform centre - cups centre) px
    vy = {E: 63.0, F: -63.0, G: -63.0, H: 63.0}
    hlin, hb = {E: 63.0, F: 63.0}, -62.0          # platform height px
    wlin, wb = {A: 63.0, B: 63.0}, -62.0          # platform width px
    hw = {A: 63.0, B: 63.0, E: -63.0, F: -63.0}   # width - height px

    def step_pair(idx, lin, bias, off0, off1):
        for j, off in ((idx, off0), (idx + 1, off1)):
            for k, v in lin.items():
                Wz[j, k] += 2.0 * v
            bz[j] = 2.0 * bias - off

    step_pair(0, add(vx, hlin, -1.5), -1.5 * hb, 0.0, 1.0)        # need left
    step_pair(2, add(neg(vx), hlin, -1.5), -1.5 * hb, 0.0, 1.0)   # need right
    step_pair(4, add(vy, wlin, -1.5), -1.5 * wb, 0.0, 1.0)        # need up
    step_pair(6, add(neg(vy), wlin, -1.5), -1.5 * wb, 0.0, 1.0)   # need down
    step_pair(8, hw, 0.0, 0.5, 1.5)          # horizontal platform
    step_pair(10, neg(hw), 0.0, 0.5, 1.5)    # vertical platform

    Wmz, bm = p["Wmz"], p["bm"]
    pairs = [((0, 1), (8, 9)), ((2, 3), (8, 9)), ((4, 5), (10, 11)), ((6, 7), (10, 11))]
    for m, ((a1, a2), (o1, o2)) in enumerate(pairs):
        Wmz[m, a1] = 6.0; Wmz[m, a2] = -6.0
        Wmz[m, o1] = 6.0; Wmz[m, o2] = -6.0
        bm[m] = -9.0

    Wo, bo = p["Wo"], p["bo"]
    # m0 -> ACTION3 (left), m1 -> ACTION4 (right), m2 -> ACTION1 (up), m3 -> ACTION2 (down)
    for m, act in [(0, 3), (1, 4), (2, 1), (3, 2)]:
        Wo[act, 128 + m] = 10.0
        bo[act] = 10.0
    bo[5] = 5.0      # spill once aligned
    bo[0] = -20.0    # never RESET
    bo[6] = -20.0    # never click
    bo[7] = -20.0
    return p

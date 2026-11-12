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

    # ---- conv features: colour-1 pixels in arena (rows < 54) ----
    C = 2
    p["W1"][0, 1, C, C] = 1.0; p["W1"][0, 17, C, C] = 1.0; p["b1"][0] = -1.0
    p["W1"][1, 1, C, C] = 1.0; p["W1"][1, 16, C, C] = 1.0; p["b1"][1] = -1.0
    p["W1"][2, 1, C, C] = 1.0; p["W1"][2, 17, C, C] = -1.0
    p["W1"][3, 1, C, C] = 1.0; p["W1"][3, 16, C, C] = -1.0
    p["W1"][4, 16, C, C] = 126.0; p["b1"][4] = -107.0
    for c in range(4):
        p["W2"][c, c, 1, 1] = 1.0
        p["W2"][c, 4, 1, 1] = -2.0
        p["W3"][c, c] = 1.0

    # ---- thermometer codes of s1, s2 ----
    N = 13
    scal = [((0, 1), 30.5), ((2, 3), 36.5)]
    for j, (chs, t0) in enumerate(scal):
        for k in range(N):
            t = t0 + 6.0 * k
            za = j * 2 * N + 2 * k
            zb = za + 1
            for ch in chs:
                p["Wz"][za, 32 + ch] = 63.0
                p["Wz"][zb, 32 + ch] = 63.0
            p["bz"][za] = -t + 0.5
            p["bz"][zb] = -t - 0.5
            mb = j * N + k
            p["Wmz"][mb, za] = 10.0
            p["Wmz"][mb, zb] = -10.0
            p["bm"][mb] = -5.0

    # ---- effect flags and prev-action copies ----
    A, B = 10.0, 5.0
    FL, PV, PL = 2 * N, 2 * N + 16, 2 * N + 20   # 26, 42, 46
    for ai in range(4):
        a = ai + 1
        for j in range(2):
            for si, s in enumerate((1.0, -1.0)):
                u = FL + ai * 4 + j * 2 + si
                for k in range(N):
                    za = j * 2 * N + 2 * k
                    p["Wmz"][u, za] += 8.0 * s
                    p["Wmz"][u, za + 1] += -8.0 * s
                    p["Wm"][u, j * N + k] = -4.0 * s
                p["bm"][u] = -4.0 * s * N - 204.0
                p["Wma"][u, a] = 200.0
                p["Wo"][a, 128 + u] = A / 2.0
                p["bo"][a] += A / 2.0
        p["Wma"][PV + ai, a] = 8.0
        p["bm"][PV + ai] = -4.0
    prev_of = {1: 4, 3: 1, 2: 3, 4: 2}
    for a in range(1, 5):
        b = prev_of[a]
        p["Wo"][a, 128 + PV + (b - 1)] = B / 2.0
        p["bo"][a] += B / 2.0
    p["bo"][4] += 1.0

    p["bo"][0] = -100.0
    p["bo"][5] = -100.0
    p["bo"][6] = -100.0
    p["bo"][7] = -100.0

    # ---- open-loop plan: RESET, then level-1 solution (pulse + shift register) ----
    plan = [0, 1, 1, 1, 4, 4, 4, 4, 3, 3, 2, 2, 4, 4, 3, 1, 4]
    assert PL + len(plan) <= 64
    p["Wma"][PL, :] = -8.0
    p["bm"][PL] = 4.0
    for k in range(1, len(plan)):
        p["Wm"][PL + k, PL + k - 1] = 6.0
        p["bm"][PL + k] = -3.0
    for k, a in enumerate(plan):
        W = 300.0 if a == 0 else 50.0
        p["Wo"][a, 128 + PL + k] += W / 2.0
        p["bo"][a] += W / 2.0
    return p

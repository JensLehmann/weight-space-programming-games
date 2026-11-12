import numpy as np

GL = ("..#.. ..#.. .###. ..#.. #####|....# ..#.# ##### #.#.. #....|##### #.#.. #.... #.#.. #####|"
      "..#.. ##### #.#.# #.#.# ..#..|#...# ##### ..#.. ##### #...#|##### ..#.. ###.. #.... #####|"
      "#...# ##### #...# #...# ##.##|"
      "#.... ####. #..#. #..#. #####|##### #...# ###.# #.#.# #####|..### ..#.# ##### #.#.. ###..|"
      ".###. .#.#. ##### #...# #####|##### #...# #...# ##### ..#..|####. #..## #...# ##..# .####|"
      "..#.. ##### #.#.# ##### ..#..|"
      "#.#.# #.... #.#.# #.... #####|###.# #.... ..... ....# #.###|#.### ..#.. ..#.. ..#.. ###.#|"
      "..#.. #.#.# ..#.. #.#.# ..#..|##.## #...# #.#.# #...# ##.##|#.### #.#.# ###.# ..... #.#.#|"
      "#...# ..... #...# ##.## .#.#.")


def build():
    temps = [np.array([[1.0 if ch == "#" else 0.0 for ch in row] for row in g.split()])
             for g in GL.split("|")]
    NG = len(temps)
    chans = []
    for gi, t in enumerate(temps):
        seen = []
        for k in range(4):
            r = np.rot90(t, k)
            if not any(np.array_equal(r, s) for s in seen):
                seen.append(r)
                chans.append((gi, r))
    K = len(chans)
    Wg = np.zeros((K, 1, 5, 5)); bgv = np.zeros(K); Wsum = np.zeros((NG, K, 1, 1))
    for c, (gi, r) in enumerate(chans):
        Wg[c, 0] = 2 * r - 1
        bgv[c] = -(r.sum() - 1)
        Wsum[gi, c, 0, 0] = 1
    Wt = np.zeros((1, 16, 1, 1)); Wt[0, 6:16, 0, 0] = 1
    Wrc = np.zeros((1, 1, 7, 7)); Wrc[0, 0, 0, 3] = 1; Wrc[0, 0, 6, 3] = 1
    brc = np.array([-1.0])
    Wmb = np.zeros((1, 1, 7, 7)); Wmb[0, 0, 0, 1:6] = 1
    Wmb[0, 0, 1, 1] = 1; Wmb[0, 0, 1, 5] = 1; Wmb[0, 0, 1, 3] = -5
    bmb = np.array([-6.0])
    Wdn = np.zeros((1, 1, 7, 7)); Wdn[0, 0, 0, 3] = 1
    Wup = np.zeros((1, 1, 7, 7)); Wup[0, 0, 6, 3] = 1
    Wband = np.zeros((1, 1, 7, 7)); Wband[0, 0, :, 3] = 1
    Wrule = np.zeros((1, 16, 7, 7)); Wrule[0, 3, 3, 4:7] = 1; Wrule[0, 2, 0, 5] = 1
    brule = np.array([-3.0])
    Wleft = np.zeros((1, 1, 7, 7)); Wleft[0, 0, 3, 6] = 1
    Wright = np.zeros((1, 1, 7, 7)); Wright[0, 0, 3, 0] = 1
    Wr1 = np.zeros((1, 1, 3, 3)); Wr1[0, 0, 1, 0] = 1
    Wone = np.ones((1, NG, 1, 1))
    A2 = np.zeros((NG, NG)); A1 = np.zeros((NG, NG))
    for h in range(NG):
        for g in range(NG):
            if h // 7 == g // 7:
                d = (h - g) % 7
                if d in (1, 2, 3):
                    A2[h, g] = 1
                elif d in (4, 5, 6):
                    A1[h, g] = 1
    Wout = np.zeros((8, 2 * NG)); Wout[2, :NG] = 2; Wout[1, NG:] = 2
    bout = np.array([-9.0, 0, 0, -1, 1, -9, -9, -9])

    def cv(i, x, w, b=None):
        d = {"id": i, "op": "conv", "in": [x], "w": w}
        if b:
            d["b"] = b
        return d

    def op(i, o, *ins):
        return {"id": i, "op": o, "in": list(ins)}

    graph = [
        {"id": "s5", "op": "slice", "in": ["x0"], "start": 5, "stop": 6},
        cv("gr0", "s5", "Wg", "bgv"), op("gr", "relu", "gr0"), cv("G", "gr", "Wsum"),
        cv("Tt", "x0", "Wt"), cv("rc0", "Tt", "Wrc", "brc"), op("RC", "relu", "rc0"),
        op("Gc", "mul", "G", "RC"),
        {"id": "s0", "op": "slice", "in": ["x0"], "start": 0, "stop": 1},
        cv("mb0", "s0", "Wmb", "bmb"), op("MB", "relu", "mb0"),
        cv("click", "MB", "Wdn"),
        cv("u1", "MB", "Wup"), cv("u2", "u1", "Wup"), cv("Bd", "u2", "Wband"),
        op("Lm", "mul", "Gc", "Bd"), op("L", "gmax", "Lm"),
        op("Cm", "mul", "G", "click"), op("Cur", "gmax", "Cm"),
        cv("rm0", "x0", "Wrule", "brule"), op("RM", "relu", "rm0"), cv("RMc", "RM", "Wleft"),
        op("gl", "mul", "Gc", "L"), cv("mm", "gl", "Wone"), op("m", "mul", "RMc", "mm"),
        cv("r1", "m", "Wright"), cv("r2", "r1", "Wright"), cv("r3", "r2", "Wright"),
        cv("r4", "r3", "Wr1"),
        op("tg0", "mul", "Gc", "r4"), op("Tg", "gmax", "tg0"),
        {"id": "v2", "op": "dense", "in": ["Cur"], "w": "A2"},
        op("f2", "vmul", "v2", "Tg"),
        {"id": "v1", "op": "dense", "in": ["Cur"], "w": "A1"},
        op("f1", "vmul", "v1", "Tg"),
        op("ff", "vconcat", "f2", "f1"),
        {"id": "logits", "op": "dense", "in": ["ff"], "w": "Wout", "b": "bout"},
    ]
    params = {"Wg": Wg, "bgv": bgv, "Wsum": Wsum, "Wt": Wt, "Wrc": Wrc, "brc": brc,
              "Wmb": Wmb, "bmb": bmb, "Wdn": Wdn, "Wup": Wup, "Wband": Wband,
              "Wrule": Wrule, "brule": brule, "Wleft": Wleft, "Wright": Wright, "Wr1": Wr1,
              "Wone": Wone, "A2": A2, "A1": A1, "Wout": Wout, "bout": bout}
    params = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    return {"graph": graph, "memory_size": 0, "params": params}

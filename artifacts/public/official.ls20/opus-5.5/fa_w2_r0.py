import numpy as np

def build():
    P = {}
    g = []
    def N(i, op, ins, **k):
        d = {"id": i, "op": op, "in": ins}; d.update(k); g.append(d)
    K = 4000.0
    w = np.zeros((2, 3, 1, 1)); b = np.zeros(2)
    w[0, 0, 0, 0] = -K; b[0] = K * 52.5 / 63
    w[1, 1, 0, 0] = K;  b[1] = -K * 3.5 / 63
    P["Wpm"] = w; P["bpm"] = b
    N("pmc", "conv", ["coords"], w="Wpm", b="bpm")
    N("pms", "sigmoid", ["pmc"])
    N("pma", "slice", ["pms"], start=0, stop=1)
    N("pmb", "slice", ["pms"], start=1, stop=2)
    N("pm", "mul", ["pma", "pmb"])
    N("wall", "slice", ["x0"], start=4, stop=5)
    N("five", "slice", ["x0"], start=5, stop=6)
    N("zero", "slice", ["x0"], start=0, stop=1)
    P["Wps"] = np.full((1, 1, 1, 1), -0.99); P["bps"] = np.array([0.99])
    N("pas0", "conv", ["wall"], w="Wps", b="bps")
    N("passd", "mul", ["pas0", "pm"])
    N("goal", "mul", ["five", "pm"])
    N("mod0", "mul", ["zero", "pm"])
    w = np.zeros((1, 16, 5, 5)); w[0, 12, 2, 2] = 1; w[0, 9, 4, 2] = 1
    P["Wpl"] = w; P["bpl"] = np.array([-1.0])
    N("plc", "conv", ["x0"], w="Wpl", b="bpl")
    N("plr", "relu", ["plc"])
    N("Pm", "mul", ["plr", "pm"])
    N("rc2", "slice", ["coords"], start=0, stop=2)
    N("prc", "mul", ["rc2", "Pm"])
    N("pos", "gmax", ["prc"])
    N("cnt", "gmean", ["mod0"])
    N("pm6", "vconcat", ["pos", "mem"])
    w = np.zeros((4, 6))
    w[0, 0] = 1; w[0, 2] = -1; w[1, 1] = 1; w[1, 3] = -1
    w[2] = -w[0]; w[3] = -w[1]
    P["Wd"] = w
    N("dd", "dense", ["pm6"], w="Wd")
    N("ddr", "vrelu", ["dd"])
    P["Wb"] = np.full((1, 4), -2000.0); P["bb"] = np.array([40.0])
    N("bl", "dense", ["ddr"], w="Wb", b="bb")
    N("bumped", "vsigmoid", ["bl"])
    N("cm5", "vconcat", ["cnt", "mem"])
    kk = 4096.0 * 20
    w = np.zeros((1, 5)); w[0, 0] = -kk; w[0, 4] = kk
    P["Wo"] = w; P["bo"] = np.array([-10.0])
    N("ol", "dense", ["cm5"], w="Wo", b="bo")
    N("onmod", "vsigmoid", ["ol"])
    N("mode", "vslice", ["mem"], start=2, stop=3)
    N("mb", "vmul", ["mode", "bumped"])
    N("mo", "vmul", ["mode", "onmod"])
    N("mt", "vconcat", ["mode", "bumped", "mb", "mo"])
    P["Wmd"] = np.array([[1.0, 1.0, -1.0, -1.0]])
    N("modenext", "dense", ["mt"], w="Wmd")
    N("cmax", "vslice", ["mem"], start=3, stop=4)
    N("cc", "vconcat", ["cnt", "cmax"])
    P["Wcc"] = np.array([[1.0, -1.0]])
    N("ccd", "dense", ["cc"], w="Wcc")
    N("ccr", "vrelu", ["ccd"])
    N("cmaxn", "vadd", ["cmax", "ccr"])
    N("mem_next", "vconcat", ["pos", "modenext", "cmaxn"])
    N("mdiff", "sub", ["mod0", "goal"])
    N("msel", "mul", ["mdiff", "modenext"])
    N("tgt", "add", ["goal", "msel"])
    w = np.zeros((4, 1, 3, 3))
    w[0, 0, 0, 1] = 1; w[1, 0, 2, 1] = 1; w[2, 0, 1, 0] = 1; w[3, 0, 1, 2] = 1
    P["Wsh"] = w
    body = [
        {"id": "s4", "op": "conv", "in": ["f"], "w": "Wsh"},
        {"id": "s0", "op": "slice", "in": ["s4"], "start": 0, "stop": 1},
        {"id": "s1", "op": "slice", "in": ["s4"], "start": 1, "stop": 2},
        {"id": "s2", "op": "slice", "in": ["s4"], "start": 2, "stop": 3},
        {"id": "s3", "op": "slice", "in": ["s4"], "start": 3, "stop": 4},
        {"id": "ma", "op": "max2", "in": ["s0", "s1"]},
        {"id": "mb2", "op": "max2", "in": ["s2", "s3"]},
        {"id": "mm", "op": "max2", "in": ["ma", "mb2"]},
        {"id": "mp", "op": "mul", "in": ["mm", "passd"]},
        {"id": "fn", "op": "max2", "in": ["mp", "tgt"]},
    ]
    g.append({"id": "L", "op": "loop", "T": 256,
              "state": [{"id": "f", "zeros": [1, 64, 64]}],
              "body": body, "next": {"f": "fn"}})
    w = np.zeros((4, 1, 5, 5))
    w[0, 0, 0, 2] = 1; w[1, 0, 4, 2] = 1; w[2, 0, 2, 0] = 1; w[3, 0, 2, 4] = 1
    P["Wd5"] = w
    N("d5", "conv", ["L.f"], w="Wd5")
    w = np.zeros((4, 4, 7, 7))
    w[0, 0, 0, 3] = 1; w[1, 1, 6, 3] = 1; w[2, 2, 3, 0] = 1; w[3, 3, 3, 6] = 1
    P["Wd7"] = w
    N("d7", "conv", ["d5"], w="Wd7")
    N("dv", "mul", ["d7", "Pm"])
    N("vals", "gmax", ["dv"])
    w = np.zeros((8, 4))
    for k in range(4):
        w[1 + k, k] = 100.0
    P["Wl"] = w; P["bl"] = np.array([0, 1e-4, 0, 0, 0, 0, 0, 0], dtype=float)
    N("logits", "dense", ["vals"], w="Wl", b="bl")
    P["Wc"] = np.zeros((1, 3, 1, 1))
    N("click", "conv", ["coords"], w="Wc")
    params = {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}
    return {"graph": g, "memory_size": 4, "params": params}

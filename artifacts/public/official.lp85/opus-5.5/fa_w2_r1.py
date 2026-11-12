import numpy as np

def build():
    P = {}
    def z(*s): return np.zeros(s, np.float32)

    WD = z(4, 2, 7, 7)
    s1c = [(-1,-1),(-1,2),(2,-1),(2,2)]
    s1e = [(-1,0),(-1,1),(2,0),(2,1),(0,-1),(1,-1),(0,2),(1,2)]
    s2c = [(-2,-2),(-2,3),(3,-2),(3,3)]
    s2e = [(-2,0),(-2,1),(3,0),(3,1),(0,-2),(1,-2),(0,3),(1,3)]
    for ci in range(2):
        for (o, cs, es) in [(2*ci, s1c, s1e), (2*ci+1, s2c, s2e)]:
            for (a, b) in cs: WD[o, ci, a+3, b+3] = 1
            for (a, b) in es: WD[o, ci, a+3, b+3] = -1
    P["W_D"] = WD; P["b_D"] = np.full(4, -3.0, np.float32)

    WC = z(4, 4, 7, 7)
    for ci in range(2):
        for (a, b) in s1c: WC[2*ci, 2*ci, -a+3, -b+3] = 1
        for (a, b) in [(-3,-3),(-3,3),(3,-3),(3,3)]: WC[2*ci+1, 2*ci+1, -a+3, -b+3] = 1
    P["W_C"] = WC
    WL = z(2, 4, 3, 3)
    for ci in range(2):
        WL[ci, 2*ci, 1, 1] = 1
        for (a, b) in [(0,0),(-1,0),(0,-1),(-1,-1)]: WL[ci, 2*ci+1, a+1, b+1] = 1
    P["W_L"] = WL
    WT = z(2, 4, 1, 1)
    WT[0, 0] = WT[0, 1] = WT[1, 2] = WT[1, 3] = 1
    P["W_T"] = WT
    WR = z(2, 2, 3, 3); WR[0, 0] = 1; WR[1, 1] = 1
    P["W_R"] = WR
    P["W_E"] = np.array([[4096.0, 4096.0]], np.float32)

    WB = z(2, 16, 5, 5)
    for o, c in [(0, 8), (1, 14)]:
        WB[o, c, 2, 2] = 1; WB[o, c, 2, 1] = -1; WB[o, c, 1, 2] = -1; WB[o, c, 1, 4] = -1
    P["W_B"] = WB
    P["W_S"] = np.ones((1, 2, 1, 1), np.float32)
    WM = z(2, 3, 1, 1)
    WM[0, 0, 0, 0] = 126; WM[0, 2, 0, 0] = -5
    WM[1, 1, 0, 0] = 126; WM[1, 2, 0, 0] = -1
    P["W_M"] = WM
    WK = z(1, 3, 1, 1); WK[0, 0, 0, 0] = 63; WK[0, 1, 0, 0] = 4032
    P["W_K"] = WK
    P["W_2"] = np.full((1, 1, 1, 1), 2.0, np.float32); P["b_2"] = np.array([-1.0], np.float32)
    P["W_G"] = np.full((1, 1, 1, 1), -1.0, np.float32); P["b_G"] = np.array([5000.0], np.float32)
    P["Wn"] = np.array([[-1.0]], np.float32); P["bo1"] = np.array([1.0], np.float32)
    P["W_1"] = np.array([[1.0]], np.float32)
    P["b_m1"] = np.array([-1.0], np.float32)
    P["b_ca"] = np.array([-4.0], np.float32)
    P["W_12"] = np.array([[1.0, -1.0]], np.float32)
    P["W_dd"] = np.array([[0.0, 2.0, 0.0, 0.0, -2.0]], np.float32)
    P["W_df"] = np.array([[-1.0, 0.0, 0.0, 0.0, -1.0]], np.float32)
    P["b_df"] = np.array([5000.0], np.float32)
    P["W_z0"] = np.array([[0.0]], np.float32)
    P["W_lo"] = z(8, 4)
    blo = z(8); blo[6] = 1.0
    P["b_lo"] = blo

    def n(i, op, ins, **kw):
        d = {"id": i, "op": op, "in": ins}; d.update(kw); return d

    g = [
        n("xg", "slice", ["x0"], start=11, stop=13),
        n("dc", "conv", ["xg"], w="W_D", b="b_D"),
        n("A", "relu", ["dc"]),
        n("co", "conv", ["A"], w="W_C"),
        n("dl", "conv", ["co"], w="W_L"),
        n("sm", "sub", ["xg", "dl"]),
        n("S", "relu", ["sm"]),
        n("Tm", "conv", ["A"], w="W_T"),
        n("ones", "slice", ["coords"], start=2, stop=3),
        {"id": "D", "op": "loop", "T": 64,
         "state": [{"id": "r", "init": "S"}, {"id": "acc", "zeros": [2, 64, 64]}],
         "body": [n("rc", "conv", ["r"], w="W_R"),
                  n("rn", "min2", ["rc", "ones"]),
                  n("om", "sub", ["ones", "rn"]),
                  n("an", "add", ["acc", "om"])],
         "next": {"r": "rn", "acc": "an"}},
        n("pm", "mul", ["D.acc", "Tm"]),
        n("pg", "gmean", ["pm"]),
        n("e", "dense", ["pg"], w="W_E"),
        n("rc5", "conv", ["x0"], w="W_B"),
        n("rr", "relu", ["rc5"]),
        n("bs", "conv", ["rr"], w="W_S"),
        n("mk", "conv", ["coords"], w="W_M"),
        n("mr", "relu", ["mk"]),
        n("mm", "min2", ["mr", "ones"]),
        n("m0", "slice", ["mm"], start=0, stop=1),
        n("m1", "slice", ["mm"], start=1, stop=2),
        n("mk2", "min2", ["m0", "m1"]),
        n("B", "min2", ["bs", "mk2"]),
        n("K", "conv", ["coords"], w="W_K"),
        n("kc", "vslice", ["mem"], start=0, stop=1),
        n("Kd", "sub", ["K", "kc"]),
        n("st1", "conv", ["Kd"], w="W_2", b="b_2"),
        n("st2", "relu", ["st1"]),
        n("step", "min2", ["st2", "ones"]),
        n("Gk", "conv", ["K"], w="W_G", b="b_G"),
        n("G", "mul", ["B", "Gk"]),
        n("v2", "gmax", ["G"]),
        n("Gc", "mul", ["G", "step"]),
        n("v1", "gmax", ["Gc"]),
        n("fa", "dense", ["v1"], w="Wn", b="bo1"),
        n("fb", "vrelu", ["fa"]),
        n("f", "dense", ["fb"], w="Wn", b="bo1"),
        n("v12", "vconcat", ["v1", "v2"]),
        n("d12", "dense", ["v12"], w="W_12"),
        n("fm", "vmul", ["f", "d12"]),
        n("nv", "vadd", ["v2", "fm"]),
        n("me", "vconcat", ["mem", "e"]),
        # imp: e <= e_prev
        n("dd", "dense", ["me"], w="W_dd", b="bo1"),
        n("r1", "vrelu", ["dd"]),
        n("t1", "dense", ["r1"], w="Wn", b="bo1"),
        n("t2", "vrelu", ["t1"]),
        n("imp", "dense", ["t2"], w="Wn", b="bo1"),
        # strict: e <= e_prev - 1
        n("dd2", "dense", ["me"], w="W_dd", b="b_m1"),
        n("r2", "vrelu", ["dd2"]),
        n("u1", "dense", ["r2"], w="Wn", b="bo1"),
        n("u2", "vrelu", ["u1"]),
        n("strict", "dense", ["u2"], w="Wn", b="bo1"),
        n("stt", "vslice", ["mem"], start=2, stop=3),
        n("isk", "vmul", ["imp", "stt"]),
        # tie cap
        n("cn", "vslice", ["mem"], start=3, stop=4),
        n("ca", "dense", ["cn"], w="W_1", b="b_ca"),
        n("car", "vrelu", ["ca"]),
        n("nb", "dense", ["car"], w="Wn", b="bo1"),
        n("nc", "vrelu", ["nb"]),
        n("ns", "vadd", ["nc", "strict"]),
        n("nsa", "dense", ["ns"], w="Wn", b="bo1"),
        n("nsb", "vrelu", ["nsa"]),
        n("ncs", "dense", ["nsb"], w="Wn", b="bo1"),
        n("keep", "vmul", ["isk", "ncs"]),
        n("sw", "dense", ["keep"], w="Wn", b="bo1"),
        n("notstr", "dense", ["strict"], w="Wn", b="bo1"),
        n("c1", "dense", ["cn"], w="W_1", b="bo1"),
        n("c2", "vmul", ["c1", "notstr"]),
        n("cn2", "vmul", ["c2", "keep"]),
        n("mn", "vconcat", ["mem", "nv"]),
        n("df", "dense", ["mn"], w="W_df", b="b_df"),
        n("sd", "vmul", ["sw", "df"]),
        n("ks", "vadd", ["kc", "sd"]),
        n("on", "dense", ["e"], w="W_z0", b="bo1"),
        n("mem_next", "vconcat", ["ks", "e", "on", "cn2"]),
        n("logits", "dense", ["mem_next"], w="W_lo", b="b_lo"),
        n("Kc", "sub", ["K", "ks"]),
        n("Ka", "abs", ["Kc"]),
        n("Ks", "conv", ["Ka"], w="W_G", b="b_G"),
        n("click", "mul", ["B", "Ks"]),
    ]
    return {"graph": g, "memory_size": 4, "params": P}

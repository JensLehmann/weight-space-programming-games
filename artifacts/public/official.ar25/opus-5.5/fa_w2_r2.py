import numpy as np

def build():
    P = {}
    G = []
    def node(id_, op, ins, **kw):
        d = {"id": id_, "op": op, "in": ins}; d.update(kw); G.append(d); return id_

    w = np.zeros((3, 3, 1, 1)); w[0, 0] = -1260; w[0, 2] = 1250
    w[1, 1] = -1260; w[1, 2] = 1250; w[2, 2] = 1
    P["Wmk"] = w
    node("mk3", "conv", ["coords"], w="Wmk")
    node("mk3r", "relu", ["mk3"])
    node("mka", "slice", ["mk3r"], start=0, stop=1)
    node("mkb", "slice", ["mk3r"], start=1, stop=2)
    node("mkc", "slice", ["mk3r"], start=2, stop=3)
    node("mkab", "min2", ["mka", "mkb"])
    node("msk", "min2", ["mkab", "mkc"])
    w = np.zeros((1, 16, 1, 1)); w[0, 4] = 128; w[0, 5] = 128; P["WI"] = w
    w = np.zeros((1, 16, 1, 1)); w[0, 11] = 1; P["WB"] = w
    node("Iraw", "conv", ["x0"], w="WI")
    node("F0", "mul", ["Iraw", "msk"])
    node("Braw", "conv", ["x0"], w="WB")
    node("Bm", "mul", ["Braw", "msk"])
    w4 = np.zeros((4, 1, 3, 3))
    w4[0, 0, 0, 1] = 1; w4[1, 0, 2, 1] = 1; w4[2, 0, 1, 0] = 1; w4[3, 0, 1, 2] = 1
    P["W4"] = w4; P["b4"] = -np.ones(4)
    body = [
        {"id": "n4", "op": "conv", "in": ["F"], "w": "W4", "b": "b4"},
        {"id": "q0", "op": "slice", "in": ["n4"], "start": 0, "stop": 1},
        {"id": "q1", "op": "slice", "in": ["n4"], "start": 1, "stop": 2},
        {"id": "q2", "op": "slice", "in": ["n4"], "start": 2, "stop": 3},
        {"id": "q3", "op": "slice", "in": ["n4"], "start": 3, "stop": 4},
        {"id": "qa", "op": "max2", "in": ["q0", "q1"]},
        {"id": "qb", "op": "max2", "in": ["q2", "q3"]},
        {"id": "qm", "op": "max2", "in": ["qa", "qb"]},
        {"id": "Fn", "op": "max2", "in": ["qm", "F0"]},
    ]
    G.append({"id": "D", "op": "loop", "T": 128,
              "state": [{"id": "F", "init": "F0"}], "body": body, "next": {"F": "Fn"}})
    P["Wone"] = np.ones((1, 1, 1, 1)); P["bm128"] = np.array([-128.0])
    node("Fm", "conv", ["D.F"], w="Wone", b="bm128")
    node("prod", "mul", ["Bm", "Fm"])
    node("s", "gmean", ["prod"])
    # motion signature
    node("cr", "slice", ["coords"], start=0, stop=1)
    node("cc", "slice", ["coords"], start=1, stop=2)
    node("rr", "mul", ["cr", "cr"])
    node("c2", "mul", ["cc", "cc"])
    node("rc", "concat", ["rr", "c2"])
    w = np.zeros((1, 2, 1, 1)); w[0, 0] = 1.0; w[0, 1] = 0.731; P["Whq"] = w
    node("hq", "conv", ["rc"], w="Whq")
    node("hp", "mul", ["F0", "hq"])
    node("h", "gmean", ["hp"])

    N = 20
    node("v", "vconcat", ["s", "mem", "aprev", "h"])
    w = np.zeros((2, N)); w[0, 0] = 8192; w[0, 1] = -8192; w[1, 0] = -8192; w[1, 1] = 8192
    P["Wd1"] = w; P["bd1"] = -np.ones(2)
    node("d1", "dense", ["v"], w="Wd1", b="bd1")
    node("u", "vrelu", ["d1"])
    P["WnI2"] = -np.eye(2); P["one2"] = np.ones(2)
    node("d2", "dense", ["u"], w="WnI2", b="one2")
    node("d2r", "vrelu", ["d2"])
    node("GW", "dense", ["d2r"], w="WnI2", b="one2")
    w = np.zeros((1, N)); w[0, 12:16] = 1; P["Wmv"] = w
    node("mv", "dense", ["v"], w="Wmv")
    node("gp", "vslice", ["GW"], start=0, stop=1)
    node("wn", "vslice", ["GW"], start=1, stop=2)
    node("g", "vmul", ["gp", "mv"])
    node("wf", "vmul", ["wn", "mv"])
    P["Wneg1"] = -np.ones((1, 1)); P["one1"] = np.ones(1)
    node("omg", "dense", ["g"], w="Wneg1", b="one1")
    # moved flag
    w = np.zeros((2, N)); w[0, 19] = 1000; w[0, 10] = -1000; w[1, 19] = -1000; w[1, 10] = 1000
    P["Wdh"] = w
    node("dh1", "dense", ["v"], w="Wdh", b="bd1")
    node("dhu", "vrelu", ["dh1"])
    P["Wneg2"] = -np.ones((1, 2))
    node("dh2", "dense", ["dhu"], w="Wneg2", b="one1")
    node("dh2r", "vrelu", ["dh2"])
    node("moved", "dense", ["dh2r"], w="Wneg1", b="one1")
    node("qn", "dense", ["GW"], w="Wneg2", b="one1")
    node("pl0", "vmul", ["qn", "mv"])
    node("pl", "vmul", ["pl0", "moved"])
    node("omp", "dense", ["pl"], w="Wneg1", b="one1")
    node("gg", "vadd", ["g", "pl"])
    node("omgg", "dense", ["gg"], w="Wneg1", b="one1")
    w = np.zeros((1, N)); w[0, 1] = 1; w[0, 0] = -1; P["Wdf"] = w
    node("diff", "dense", ["v"], w="Wdf")
    node("wd", "vmul", ["wf", "diff"])
    node("s0", "vslice", ["v"], start=0, stop=1)
    node("sref_n", "vadd", ["s0", "wd"])
    node("trd", "vslice", ["v"], start=2, stop=6)
    node("a14", "vslice", ["v"], start=12, stop=16)
    node("aq", "vmul", ["a14", "omp"])
    node("ta", "vadd", ["trd", "aq"])
    P["WnI4"] = -np.eye(4); P["one4"] = np.ones(4)
    node("ta2", "dense", ["ta"], w="WnI4", b="one4")
    node("ta2r", "vrelu", ["ta2"])
    node("trc", "dense", ["ta2r"], w="WnI4", b="one4")
    w = np.zeros((1, N)); w[0, 11] = -1; w[0, 16] = -1; w[0, 17] = -1; P["Wka"] = w
    node("keepA", "dense", ["v"], w="Wka", b="one1")
    node("keep", "vmul", ["keepA", "omg"])
    node("tried_n", "vmul", ["trc", "keep"])
    node("pref", "vslice", ["v"], start=6, stop=10)
    node("p1", "vmul", ["pref", "omgg"])
    node("p2", "vmul", ["a14", "gg"])
    node("pref_n", "vadd", ["p1", "p2"])
    node("href_n", "vslice", ["v"], start=19, stop=20)
    node("unt", "dense", ["tried_n"], w="WnI4", b="one4")
    P["W5I"] = 5 * np.eye(4); P["bval"] = np.array([10.4, 10.3, 10.2, 10.1])
    node("val", "dense", ["pref_n"], w="W5I", b="bval")
    node("Lk", "vmul", ["unt", "val"])
    node("lw", "vconcat", ["Lk", "wf"])
    w = np.zeros((8, 5))
    for k in range(4):
        w[1 + k, k] = 1
    w[7, 4] = 100
    P["Wlo"] = w
    P["blo"] = np.array([-100, 0, 0, 0, 0, 1, -100, 0], dtype=np.float64)
    node("logits", "dense", ["lw"], w="Wlo", b="blo")
    node("mem_next", "vconcat", ["sref_n", "tried_n", "pref_n", "href_n"])
    node("click", "slice", ["msk"], start=0, stop=1)

    params = {k: np.asarray(v, dtype=np.float32) for k, v in P.items()}
    return {"graph": G, "memory_size": 10, "params": params}

import numpy as np

def build():
    P = {}
    Wpm = np.zeros((2, 3, 1, 1), np.float32)
    Wpm[0, 0, 0, 0] = 630.0; Wpm[0, 2, 0, 0] = -95.0
    Wpm[1, 0, 0, 0] = -630.0; Wpm[1, 2, 0, 0] = 625.0
    P["Wpm"] = Wpm
    cols = (10, 6, 15, 11, 12, 8)
    Wf = np.zeros((7, 16, 1, 1), np.float32)
    for i, c in enumerate(cols):
        Wf[i, c, 0, 0] = 1.0
    Wf[6, 9, 0, 0] = 1.0
    P["Wf"] = Wf
    def diag(k):
        w = np.zeros((6, 6, k, k), np.float32)
        for i in range(6):
            w[i, i] = 1.0
        return w
    P["D7"] = diag(7); P["D5"] = diag(5)
    P["Wsum"] = np.ones((1, 6, 1, 1), np.float32)
    P["Woth"] = (1.0 - np.eye(6, dtype=np.float32)).reshape(6, 6, 1, 1)
    P["K3"] = np.ones((1, 1, 3, 3), np.float32)
    P["K5"] = np.ones((1, 1, 5, 5), np.float32)
    P["K7"] = np.ones((1, 1, 7, 7), np.float32)
    P["b8"] = np.array([-8.0], np.float32)
    P["Ks"] = np.full((1, 1, 1, 1), 0.001, np.float32)
    P["Wl"] = np.zeros((8, 8), np.float32)
    bl = np.zeros(8, np.float32); bl[6] = 10.0
    P["bl"] = bl
    g = [
        {"id": "pmc", "op": "conv", "in": ["coords"], "w": "Wpm"},
        {"id": "pa", "op": "slice", "in": ["pmc"], "start": 0, "stop": 1},
        {"id": "pb", "op": "slice", "in": ["pmc"], "start": 1, "stop": 2},
        {"id": "pmin", "op": "min2", "in": ["pa", "pb"]},
        {"id": "pr", "op": "relu", "in": ["pmin"]},
        {"id": "one", "op": "slice", "in": ["coords"], "start": 2, "stop": 3},
        {"id": "pm", "op": "min2", "in": ["pr", "one"]},
        {"id": "fg", "op": "conv", "in": ["x0"], "w": "Wf"},
        {"id": "fr6", "op": "slice", "in": ["fg"], "start": 0, "stop": 6},
        {"id": "gr", "op": "slice", "in": ["fg"], "start": 6, "stop": 7},
        {"id": "Fc", "op": "mul", "in": ["fr6", "pm"]},
        {"id": "G", "op": "mul", "in": ["gr", "pm"]},
        {"id": "F", "op": "conv", "in": ["Fc"], "w": "Wsum"},
        {"id": "G3", "op": "conv", "in": ["G"], "w": "K3"},
        {"id": "fo0", "op": "sub", "in": ["Fc", "G3"]},
        {"id": "Fo", "op": "relu", "in": ["fo0"]},
        {"id": "n1", "op": "conv", "in": ["Fo"], "w": "D7"},
        {"id": "n1c", "op": "min2", "in": ["n1", "one"]},
        {"id": "n2", "op": "conv", "in": ["n1c"], "w": "D5"},
        {"id": "N6", "op": "min2", "in": ["n2", "one"]},
        {"id": "d1", "op": "conv", "in": ["Fc"], "w": "D7"},
        {"id": "d1c", "op": "min2", "in": ["d1", "one"]},
        {"id": "d2", "op": "conv", "in": ["d1c"], "w": "D7"},
        {"id": "d2c", "op": "min2", "in": ["d2", "one"]},
        {"id": "d3", "op": "conv", "in": ["d2c"], "w": "D5"},
        {"id": "D6", "op": "min2", "in": ["d3", "one"]},
        {"id": "Oc", "op": "conv", "in": ["D6"], "w": "Woth"},
        {"id": "v0", "op": "sub", "in": ["N6", "Oc"]},
        {"id": "V", "op": "relu", "in": ["v0"]},
        {"id": "Vs", "op": "conv", "in": ["V"], "w": "Wsum"},
        {"id": "a1", "op": "conv", "in": ["F"], "w": "K7"},
        {"id": "a1c", "op": "min2", "in": ["a1", "one"]},
        {"id": "a2", "op": "conv", "in": ["a1c"], "w": "K5"},
        {"id": "Na", "op": "min2", "in": ["a2", "one"]},
        {"id": "L", "op": "loop", "T": 64,
         "state": [{"id": "h", "init": "G"}, {"id": "s", "zeros": [1, 64, 64]},
                   {"id": "e", "init": "G"}, {"id": "q", "zeros": [1, 64, 64]}],
         "body": [
             {"id": "hc", "op": "conv", "in": ["h"], "w": "K3"},
             {"id": "hn", "op": "min2", "in": ["hc", "one"]},
             {"id": "sn", "op": "add", "in": ["s", "hn"]},
             {"id": "ec", "op": "conv", "in": ["e"], "w": "K3", "b": "b8"},
             {"id": "en", "op": "relu", "in": ["ec"]},
             {"id": "qn", "op": "add", "in": ["q", "en"]},
         ],
         "next": {"h": "hn", "s": "sn", "e": "en", "q": "qn"}},
        {"id": "Sa", "op": "add", "in": ["L.s", "G"]},
        {"id": "Sb", "op": "add", "in": ["Sa", "L.q"]},
        {"id": "S1", "op": "add", "in": ["Sb", "one"]},
        {"id": "t1", "op": "mul", "in": ["Vs", "S1"]},
        {"id": "t2", "op": "mul", "in": ["Na", "S1"]},
        {"id": "t2s", "op": "conv", "in": ["t2"], "w": "Ks"},
        {"id": "sc", "op": "add", "in": ["t1", "t2s"]},
        {"id": "click", "op": "mul", "in": ["sc", "pm"]},
        {"id": "logits", "op": "dense", "in": ["avail"], "w": "Wl", "b": "bl"},
    ]
    return {"graph": g, "memory_size": 0, "params": P}

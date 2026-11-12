import numpy as np

def act(obs, available, memory):
    o = np.asarray(obs)
    sub = o[50:, :]
    ys, xs = np.where(sub == 9)
    if len(xs) == 0:
        return ("ACTION6", 58, 46)
    sel = xs >= (xs.max() - 9)
    r = int(np.mean(ys[sel])) + 50
    c = int(np.mean(xs[sel]))
    return ("ACTION6", r, c)

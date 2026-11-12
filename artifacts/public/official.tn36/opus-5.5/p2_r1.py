import numpy as np

def _run_button(obs):
    ys, xs = np.where(obs[50:64, 32:64] == 9)
    if len(ys) < 10:
        return None
    return int(ys.mean()) + 50, int(xs.mean()) + 32

def _segments(obs, color):
    r0, r1, c0, c1 = 32, 51, 13, 63
    sub = (obs[r0:r1, c0:c1] == color)
    seen = np.zeros_like(sub)
    out = []
    H, W = sub.shape
    for r in range(H):
        for c in range(W):
            if sub[r, c] and not seen[r, c]:
                stack = [(r, c)]; seen[r, c] = True; pts = []
                while stack:
                    a, b = stack.pop(); pts.append((a, b))
                    for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                        x, y = a+da, b+db
                        if 0 <= x < H and 0 <= y < W and sub[x, y] and not seen[x, y]:
                            seen[x, y] = True; stack.append((x, y))
                if len(pts) <= 5:
                    a, b = pts[len(pts)//2]
                    out.append((a + r0, b + c0))
    return out

def act(obs, available, memory):
    obs = np.asarray(obs)
    btn = _run_button(obs)
    if btn is None:
        return ("ACTION6", 32, 32)
    q = memory.get("queue")
    if not q:
        phase = memory.get("phase", 0)
        memory["phase"] = phase + 1
        color = 1 if phase % 2 == 0 else 5
        segs = _segments(obs, color)
        segs = [(r, c) for (r, c) in segs
                if (obs[max(r-1,0):r+2, max(c-2,0):c+3] == 0).any()]
        q = [list(s) for s in segs] + [list(btn)]
        memory["queue"] = q
    r, c = memory["queue"].pop(0)
    return ("ACTION6", int(r), int(c))

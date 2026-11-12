import numpy as np

def _run_button(obs):
    sub = obs[48:, 30:]
    ys, xs = np.where(sub == 9)
    if len(ys) == 0:
        return (56, 36)
    return (int(ys.mean()) + 48, int(xs.mean()) + 30)

def _level1_bars(obs):
    clicks = []
    seen = set()
    for r in (42, 45):
        for c in range(13, 51):
            if obs[r, c] == 1:
                if r == 42:
                    start = c
                    while start - 1 >= 13 and obs[r, start - 1] == 1:
                        start -= 1
                    key = (r, start)
                else:
                    key = (r, c)
                if key not in seen:
                    seen.add(key)
                    clicks.append((r, c + (1 if r == 42 else 0)))
    return clicks

def act(obs, available, memory):
    if "plan" not in memory:
        plan = []
        level1 = obs[3, 1] == 5 and obs[8, 13] == 0 and obs[41, 13] == 0
        if level1:
            plan = _level1_bars(obs)
        memory["plan"] = plan
        memory["pi"] = 0
    plan = memory["plan"]
    pi = memory["pi"]
    memory["pi"] = pi + 1
    if pi < len(plan):
        r, c = plan[pi]
        return ("ACTION6", int(r), int(c))
    r, c = _run_button(obs)
    return ("ACTION6", int(r), int(c))

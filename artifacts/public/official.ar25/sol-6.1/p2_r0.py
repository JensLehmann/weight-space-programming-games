import math
import time
import itertools
from collections import deque
import numpy as np


DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def rotate(p):
    w, h, pts = p
    return h, w, tuple(sorted((y, w - 1 - x) for x, y in pts))


def patterns(a):
    out = [(a["w"], a["h"], tuple(a["pts"]))]
    for _ in range(3):
        out.append(rotate(out[-1]))
    return out


def footprint(p, cx, cy, W, H):
    w, h, pts = p
    x0, y0 = cx - w // 2, cy - h // 2
    return {(x0 + x, y0 + y) for x, y in pts
            if 0 <= x0 + x < W and 0 <= y0 + y < H}


def geometry(obs):
    yy, xx = np.where(obs[:, :63] != 5)
    if not len(xx):
        return 21, 21, 3, 0, 0
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    candidates = []
    for s in (5, 4, 3, 2, 1):
        if bw % s or bh % s:
            continue
        W, H = bw // s, bh // s
        if not (4 <= W <= 64 and 4 <= H <= 64):
            continue
        if min(64 // W, 64 // H) != s:
            continue
        ox, oy = (64 - W * s) // 2, (64 - H * s) // 2
        if abs(ox - x0) > 1 or abs(oy - y0) > 1:
            continue
        if s == 1:
            candidates.append((0.15, W, H, s, ox, oy))
            continue
        dx = math.ceil((64 - W * s) / 2) - ox
        dy = math.ceil((64 - H * s) / 2) - oy
        ui = set()
        if s == 3:
            ui.add((dy, dx))
        elif s == 4:
            ui = {(dy + i, dx + j) for i in (1, 2) for j in (1, 2)}
        elif s == 5:
            ui.add((dy + 2, dx + 2))
        errors = count = 0
        for y in range(H):
            for x in range(W):
                vals = [int(obs[oy + y * s + i, ox + x * s + j])
                        for i in range(s) for j in range(s)
                        if (i, j) not in ui and ox + x * s + j < 63]
                if vals:
                    hist = {}
                    for v in vals:
                        hist[v] = hist.get(v, 0) + 1
                    errors += len(vals) - max(hist.values())
                    count += len(vals)
        candidates.append((errors / max(1, count), W, H, s, ox, oy))
    if not candidates:
        return 21, 21, 3, 0, 0
    candidates.sort(key=lambda c: (c[0], -c[3]))
    return tuple(candidates[0][1:])


def read_frame(obs, geom):
    W, H, s, ox, oy = geom
    base = np.empty((H, W), dtype=int)
    center = np.empty((H, W), dtype=int)
    dx = math.ceil((64 - W * s) / 2) - ox
    dy = math.ceil((64 - H * s) / 2) - oy
    ui = set()
    if s == 3:
        ui.add((dy, dx))
        cy, cx = dy, dx
    elif s == 4:
        ui = {(dy + i, dx + j) for i in (1, 2) for j in (1, 2)}
        cy, cx = dy + 1, dx + 1
    elif s == 5:
        ui.add((dy + 2, dx + 2))
        cy, cx = dy + 2, dx + 2
    else:
        cy = cx = s // 2

    for y in range(H):
        for x in range(W):
            vals = []
            for i in range(s):
                for j in range(s):
                    py, px = oy + y * s + i, ox + x * s + j
                    if (i, j) not in ui and px < 63:
                        vals.append(int(obs[py, px]))
            if not vals:
                vals = [int(obs[oy + y * s, min(62, ox + x * s)])]
            base[y, x] = np.bincount(vals, minlength=16).argmax()
            center[y, x] = int(obs[min(63, oy + y * s + cy),
                                    min(62, ox + x * s + cx)])

    O = {(x, y) for y in range(H) for x in range(W)
         if int(base[y, x]) not in (4, 9, 10, 11)}
    G = {(x, y) for y in range(H) for x in range(W) if base[y, x] == 4}
    T = {(x, y) for y in range(H) for x in range(W)
         if center[y, x] == 11 or base[y, x] == 11}

    axes = []
    for d in ("v", "h"):
        n, length = (W, H) if d == "v" else (H, W)
        for p in range(n):
            cells = [(p, y) for y in range(H)] if d == "v" else [
                (x, p) for x in range(W)]
            visible = [q for q in cells if base[q[1], q[0]] == 10]
            if len(visible) < max(3, int(length * 0.30)):
                continue
            marks = [int(center[y, x]) for x, y in visible]
            zeros = marks.count(0)
            fixed = not any(c in (0, 9) for c in marks)
            axes.append({
                "d": d, "p": p, "fixed": fixed,
                "selected": zeros >= max(2, math.ceil(len(visible) * 0.5)),
                "zratio": zeros / len(visible),
            })

    selected_axes = [a for a in axes if a["selected"]]
    if len(selected_axes) > 1:
        winner = max(selected_axes, key=lambda a: a["zratio"])
        for a in axes:
            a["selected"] = a is winner

    # The selected axis also paints zero dots on any ordinary piece it crosses.
    # These dots must not be interpreted as piece selection.
    piece_center = center.copy()
    for a in axes:
        if not a["selected"]:
            continue
        for x, y in O:
            on_axis = x == a["p"] if a["d"] == "v" else y == a["p"]
            if on_axis and piece_center[y, x] == 0:
                piece_center[y, x] = 9

    SP = {p for p in O if piece_center[p[1], p[0]] == 0}
    return {"W": W, "H": H, "base": base, "center": piece_center,
            "raw_center": center, "O": O, "G": G, "T": T,
            "SP": SP, "axes": axes}


def actor_from_cells(cells, center):
    xs, ys = zip(*cells)
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0 + 1, max(ys) - y0 + 1
    marks = [int(center[y, x]) for x, y in cells]
    movable = any(c in (0, 9) for c in marks)
    locked = any(c in (5, 10) for c in marks) and not movable
    return {
        "w": w, "h": h,
        "pts": tuple(sorted((x - x0, y - y0) for x, y in cells)),
        "cx": x0 + w // 2, "cy": y0 + h // 2,
        "fixed": locked, "rot": None,
        "modes": ["both", "v", "h", "none"], "mode": "both",
    }


def initial_actors(data):
    cen = data["center"]

    def status(p):
        c = int(cen[p[1], p[0]])
        if c == 0:
            return 0
        if c == 9:
            return 1
        if c in (5, 10):
            return 2
        return 3

    unseen = set(data["O"])
    components = []
    while unseen:
        seed = min(unseen, key=lambda p: (p[1], p[0]))
        unseen.remove(seed)
        q, comp = [seed], {seed}
        kinds = {status(seed)} - {3}
        while q:
            x, y = q.pop()
            for p in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if p not in unseen:
                    continue
                k = status(p)
                if k != 3 and kinds and k not in kinds:
                    continue
                unseen.remove(p)
                comp.add(p)
                q.append(p)
                if k != 3:
                    kinds.add(k)
        components.append(comp)

    selected = [c for c in components if c & data["SP"]]
    if len(selected) > 1:
        merged = set().union(*selected)
        components = [c for c in components if not c & data["SP"]] + [merged]
    components.sort(key=lambda c: (min(y for x, y in c), min(x for x, y in c)))
    return [actor_from_cells(c, cen) for c in components if c]


def match_axes(old, fresh):
    if len(old) != len(fresh):
        return fresh, True
    remaining, result, changed = list(fresh), [], False
    for a in old:
        choices = [(abs(a["p"] - b["p"]), i)
                   for i, b in enumerate(remaining) if a["d"] == b["d"]]
        if not choices:
            return fresh, True
        _, i = min(choices)
        b = remaining.pop(i)
        changed |= a["fixed"] != b["fixed"]
        result.append(b)
    return result, changed


def update_state(data, old, last):
    W, H = data["W"], data["H"]
    if old is None or old["W"] != W or old["H"] != H:
        actors, axes = initial_actors(data), data["axes"]
        changed = True
    else:
        axes, changed = match_axes(old["axes"], data["axes"])
        oldactors = old["actors"]
        moved, dx, dy = None, 0, 0
        if last and last.get("kind") == "move":
            dx, dy = DIRS[last["action"]]
            if last.get("who", (None,))[0] == "s":
                moved = last["who"][1]
        axis_changed = {
            d: any(a["d"] == d and a["p"] != b["p"]
                   for a, b in zip(old["axes"], axes))
            for d in ("v", "h")
        }

        options = []
        for i, a in enumerate(oldactors):
            pats = patterns(a)
            centers = [(a["cx"], a["cy"])]
            if moved == i:
                centers.insert(0, (a["cx"] + dx, a["cy"] + dy))
            relevant = axis_changed["v"] or axis_changed["h"] or moved == i
            choices = []
            for cx, cy in centers:
                for k in ((0, 1) if relevant else (0,)):
                    choices.append((cx, cy, k, pats[k],
                                    footprint(pats[k], cx, cy, W, H)))
            options.append(choices)

        best, besterr = None, float("inf")
        if math.prod(len(x) for x in options) <= 4096:
            for combo in itertools.product(*options):
                union = set()
                for c in combo:
                    union |= c[4]
                err = len(union ^ data["O"])
                if err < besterr:
                    besterr, best = err, combo
                    if err == 0:
                        break

        if best is not None and besterr == 0:
            actors = []
            for i, (a, item) in enumerate(zip(oldactors, best)):
                cx, cy, k, pat, fp = item
                b = dict(a)
                b.update(cx=cx, cy=cy, w=pat[0], h=pat[1], pts=pat[2])
                if k:
                    trigger = None
                    if axis_changed["v"] or (moved == i and cx != a["cx"]):
                        trigger = "v"
                    elif axis_changed["h"] or (moved == i and cy != a["cy"]):
                        trigger = "h"
                    if trigger and b["rot"] != trigger:
                        b["rot"] = trigger
                        changed = True
                actors.append(b)
        else:
            actors = initial_actors(data)
            changed = True
            unused = set(range(len(oldactors)))
            for a in actors:
                matching = []
                for j in unused:
                    b = oldactors[j]
                    if (a["w"], a["h"], a["pts"]) in patterns(b):
                        matching.append((abs(a["cx"] - b["cx"]) +
                                         abs(a["cy"] - b["cy"]), j))
                if matching:
                    _, j = min(matching)
                    unused.remove(j)
                    b = oldactors[j]
                    a["rot"], a["modes"], a["mode"] = (
                        b["rot"], b["modes"], b["mode"])

    selected = None
    for i, a in enumerate(axes):
        if a["selected"]:
            selected = ("a", i)
            break

    if selected is None and data["SP"]:
        hits = []
        for i, a in enumerate(actors):
            fp = footprint((a["w"], a["h"], a["pts"]),
                           a["cx"], a["cy"], W, H)
            n = len(fp & data["SP"])
            if n:
                hits.append((n, i, fp))
        merge = []
        for n, i, fp in hits:
            visible = fp - data["T"]
            if visible and len(visible & data["SP"]) >= 0.8 * len(visible):
                merge.append(i)
        if len(merge) > 1:
            cells = set()
            for i in merge:
                a = actors[i]
                cells |= footprint((a["w"], a["h"], a["pts"]),
                                   a["cx"], a["cy"], W, H)
            merged = actor_from_cells(cells, data["center"])
            merged["fixed"] = False
            actors = [a for i, a in enumerate(actors) if i not in merge] + [merged]
            selected = ("s", len(actors) - 1)
            changed = True
        elif hits:
            _, i, _ = max(hits)
            selected = ("s", i)
            actors[i]["fixed"] = False

    if selected is None and old is not None:
        wanted = old.get("selected")
        if wanted and wanted[1] < len(axes if wanted[0] == "a" else actors):
            selected = wanted

    T = set(data["T"])
    if old is not None:
        T |= set(old["T"])
    return {"W": W, "H": H, "actors": actors, "axes": axes,
            "selected": selected, "T": T}, changed


def orbit(point, axes, mode):
    use = [a for a in axes if mode == "both" or mode == a["d"]]
    if not use:
        return {point}
    if (sum(a["d"] == "v" for a in use) <= 1 and
            sum(a["d"] == "h" for a in use) <= 1):
        result = {point}
        for a in use:
            result |= {(2 * a["p"] - x, y) if a["d"] == "v"
                       else (x, 2 * a["p"] - y) for x, y in list(result)}
        return result
    seen = {point}
    q = deque([(point, 0)])
    while q:
        (x, y), depth = q.popleft()
        if depth > 12:
            continue
        for a in use:
            p = (2 * a["p"] - x, y) if a["d"] == "v" else (
                x, 2 * a["p"] - y)
            if p not in seen:
                seen.add(p)
                q.append((p, depth + 1))
    return seen


def learn_modes(state, data):
    actors, axes = state["actors"], state["axes"]
    W, H = state["W"], state["H"]
    if not actors or len(actors) > 6:
        return False
    tables = []
    for a in actors:
        fp = footprint((a["w"], a["h"], a["pts"]),
                       a["cx"], a["cy"], W, H)
        variants = []
        for mode in a["modes"]:
            reflected = set()
            for p in fp:
                reflected |= orbit(p, axes, mode)
            reflected = {p for p in reflected
                         if 0 <= p[0] < W and 0 <= p[1] < H} - data["O"]
            variants.append((mode, reflected))
        tables.append(variants)
    if math.prod(len(t) for t in tables) > 4096:
        return False
    exact, best, bestscore = [], None, None
    preference = {"both": 0, "v": 1, "h": 1, "none": 2}
    for combo in itertools.product(*tables):
        union = set()
        for mode, fp in combo:
            union |= fp
        err = len(union ^ data["G"])
        score = (err, sum(preference[m] for m, fp in combo))
        if bestscore is None or score < bestscore:
            bestscore, best = score, combo
        if err == 0:
            exact.append(combo)
    changed = False
    if best is not None and bestscore[0] == 0:
        for i, a in enumerate(actors):
            mode = best[i][0]
            changed |= mode != a["mode"]
            a["mode"] = mode
            a["modes"] = [m for m in ("both", "v", "h", "none")
                          if any(c[i][0] == m for c in exact)]
    return changed


def rotation_steps(a, oldaxes, newaxes):
    d = a["rot"]
    if d is not None:
        for old, new in zip(oldaxes, newaxes):
            if old["d"] == d:
                return abs(old["p"] - new["p"])
    return 0


def reachable(a, start_k, W, H, maxdist=None, parents=False):
    pats = patterns(a)
    start = (a["cx"], a["cy"], start_k % 4)
    dist, prev = {start: 0}, {}
    q = deque([start])
    rot = a["rot"]
    while q:
        cx, cy, k = q.popleft()
        n = dist[(cx, cy, k)]
        if maxdist is not None and n >= maxdist:
            continue
        w, h, pts = pats[k]
        for action, (dx, dy) in DIRS.items():
            nx, ny = cx + dx, cy + dy
            x0, y0 = nx - w // 2, ny - h // 2
            if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
                continue
            nk = (k + 1) % 4 if (
                (rot == "v" and dx) or (rot == "h" and dy)) else k
            s = (nx, ny, nk)
            if s not in dist:
                dist[s] = n + 1
                if parents:
                    prev[s] = ((cx, cy, k), action)
                q.append(s)
    return dist, prev, pats


def mask_at(pat, cx, cy, bitmap, W, H):
    w, h, pts = pat
    x0, y0, mask = cx - w // 2, cy - h // 2, 0
    for x, y in pts:
        gx, gy = x0 + x, y0 + y
        if 0 <= gx < W and 0 <= gy < H:
            mask |= bitmap[gy * W + gx]
    return mask


def solve_options(options, missing, budget, deadline):
    n = len(options)
    unions, bybit = [], []
    for opts in options:
        u, table = 0, {}
        for j, op in enumerate(opts):
            u |= op[0]
            m = op[0]
            while m:
                b = m & -m
                table.setdefault(b, []).append(j)
                m -= b
        unions.append(u)
        bybit.append(table)
    bestcost, bestchoice = budget, None
    memo, nodes = {}, 0

    def dfs(rem, need, cost, chosen):
        nonlocal bestcost, bestchoice, nodes
        nodes += 1
        if nodes > 12000 or (
                nodes % 256 == 0 and time.perf_counter() > deadline):
            return
        if cost >= bestcost:
            return
        if not need:
            bestcost, bestchoice = cost, dict(chosen)
            return
        if not rem:
            return
        u = 0
        for i in rem:
            u |= unions[i]
        if need & ~u:
            return
        key = (rem, need)
        if memo.get(key, float("inf")) <= cost:
            return
        memo[key] = cost

        if len(rem) == 1:
            i = rem[0]
            for j, op in enumerate(options[i]):
                if cost + op[1] < bestcost and need & ~op[0] == 0:
                    bestcost = cost + op[1]
                    bestchoice = dict(chosen)
                    bestchoice[i] = j
            return

        critical, fewest = None, float("inf")
        m = need
        while m:
            b = m & -m
            count = sum(len(bybit[i].get(b, ())) for i in rem)
            if count < fewest:
                critical, fewest = b, count
                if count <= 1:
                    break
            m -= b
        if not fewest:
            return

        branches = []
        for i in rem:
            for j in bybit[i].get(critical, ()):
                op = options[i][j]
                if cost + op[1] < bestcost:
                    branches.append((op[1], -(op[0] & need).bit_count(), i, j))
        branches.sort()
        for c, _, i, j in branches:
            chosen[i] = j
            dfs(tuple(x for x in rem if x != i),
                need & ~options[i][j][0], cost + c, chosen)
            del chosen[i]

    dfs(tuple(range(n)), missing, 0, {})
    return bestcost, bestchoice


def make_plan(state, seconds=6.8):
    deadline = time.perf_counter() + seconds
    W, H = state["W"], state["H"]
    actors, axes = state["actors"], state["axes"]
    targets = sorted(state["T"])
    if not targets:
        return None
    full = (1 << len(targets)) - 1
    axis_ranges = [[a["p"]] if a["fixed"] else
                   list(range(W if a["d"] == "v" else H)) for a in axes]
    combos = list(itertools.product(*axis_ranges)) if axes else [()]
    selected = state["selected"]

    def axis_cost(values):
        cost, moving = 0, False
        for i, (a, p) in enumerate(zip(axes, values)):
            if p != a["p"]:
                moving = True
                cost += abs(p - a["p"]) + (selected != ("a", i))
        return cost, moving

    combos.sort(key=lambda v: axis_cost(v)[0])
    bestcost, bestplan = float("inf"), None
    fallback, fallbackscore = None, (-1, float("-inf"))

    for values in combos:
        if time.perf_counter() > deadline:
            break
        ac, axis_moves = axis_cost(values)
        if ac >= bestcost:
            continue
        newaxes = [dict(a, p=p) for a, p in zip(axes, values)]
        bitmaps = {}
        for mode in {a["mode"] for a in actors}:
            bitmap = [0] * (W * H)
            for i, p in enumerate(targets):
                for x, y in orbit(p, newaxes, mode):
                    if 0 <= x < W and 0 <= y < H:
                        bitmap[y * W + x] |= 1 << i
            bitmaps[mode] = bitmap

        fixedmask, movable, options, defaults = 0, [], [], {}
        for i, a in enumerate(actors):
            pats = patterns(a)
            k0 = rotation_steps(a, axes, newaxes) % 4
            bitmap = bitmaps[a["mode"]]
            defaults[i] = (a["cx"], a["cy"], pats[k0])
            if a["fixed"]:
                fixedmask |= mask_at(pats[k0], a["cx"], a["cy"], bitmap, W, H)
                continue

            opts = {}
            selection_cost = 0 if selected == ("s", i) and not axis_moves else 1
            if a["rot"]:
                cap = None if math.isinf(bestcost) else max(0, int(bestcost - ac))
                dist, _, pp = reachable(a, k0, W, H, cap)
                candidates = ((cx, cy, pp[k], d)
                              for (cx, cy, k), d in dist.items())
            else:
                pat = pats[0]
                w, h, _ = pat
                candidates = (
                    (x + w // 2, y + h // 2, pat,
                     abs(x + w // 2 - a["cx"]) +
                     abs(y + h // 2 - a["cy"]))
                    for y in range(max(0, H - h + 1))
                    for x in range(max(0, W - w + 1)))

            for cx, cy, pat, distance in candidates:
                c = distance + (selection_cost if distance else 0)
                if ac + c >= bestcost:
                    continue
                mask = mask_at(pat, cx, cy, bitmap, W, H)
                op = (mask, c, cx, cy, pat)
                if mask not in opts or c < opts[mask][1]:
                    opts[mask] = op
            if not opts:
                opts[0] = (0, 0, a["cx"], a["cy"], pats[k0])
            movable.append(i)
            options.append(sorted(opts.values(),
                                  key=lambda o: (o[1], -o[0].bit_count())))

        c, chosen = solve_options(options, full & ~fixedmask,
                                  bestcost - ac, deadline)
        if chosen is not None and ac + c < bestcost:
            goal = dict(defaults)
            for j, opindex in chosen.items():
                op = options[j][opindex]
                goal[movable[j]] = (op[2], op[3], op[4])
            bestcost = ac + c
            bestplan = {"axes": tuple(values), "goals": goal}

        if bestplan is None:
            covered, partial_cost = fixedmask, ac
            goal, unused = dict(defaults), set(range(len(options)))
            while unused:
                choice, score = None, None
                for j in unused:
                    for op in options[j]:
                        sc = ((op[0] & ~covered).bit_count(), -op[1])
                        if score is None or sc > score:
                            score, choice = sc, (j, op)
                j, op = choice
                unused.remove(j)
                covered |= op[0]
                partial_cost += op[1]
                goal[movable[j]] = (op[2], op[3], op[4])
            sc = (covered.bit_count(), -partial_cost)
            if sc > fallbackscore:
                fallbackscore = sc
                fallback = {"axes": tuple(values), "goals": goal, "partial": True}
    return bestplan if bestplan is not None else fallback


def display_point(p, geom):
    W, H, s, ox, oy = geom
    x, y = p
    return (int(min(63, max(0, oy + y * s + s // 2))),
            int(min(63, max(0, ox + x * s + s // 2))))


def click_point(who, state, data):
    kind, i = who
    W, H = state["W"], state["H"]
    if kind == "a":
        a = state["axes"][i]
        cells = [(a["p"], y) for y in range(H)] if a["d"] == "v" else [
            (x, a["p"]) for x in range(W)]
        clear = [p for p in cells if p not in data["O"]]
        intersections = set()
        for b in state["axes"]:
            if b["d"] != a["d"]:
                intersections.add((a["p"], b["p"]) if a["d"] == "v"
                                  else (b["p"], a["p"]))
        if clear:
            return next((p for p in clear if p not in intersections), clear[0])
        return cells[len(cells) // 2]
    a = state["actors"][i]
    fp = footprint((a["w"], a["h"], a["pts"]), a["cx"], a["cy"], W, H)
    visible = fp & data["O"]
    if visible:
        return min(visible, key=lambda p: (
            p in data["T"], abs(p[0] - a["cx"]) + abs(p[1] - a["cy"])))
    return min(W - 1, max(0, a["cx"])), min(H - 1, max(0, a["cy"]))


def first_move(a, goal, W, H):
    gx, gy, desired = goal
    current = (a["w"], a["h"], tuple(a["pts"]))
    if (a["cx"], a["cy"]) == (gx, gy) and current == desired:
        return None
    if not a["rot"]:
        if current != desired:
            return -1
        dx, dy = gx - a["cx"], gy - a["cy"]
        if dx:
            return 4 if dx > 0 else 3
        if dy:
            return 2 if dy > 0 else 1
        return -1
    dist, prev, pats = reachable(a, 0, W, H, parents=True)
    endings = [(d, s) for s, d in dist.items()
               if s[0] == gx and s[1] == gy and pats[s[2]] == desired]
    if not endings:
        return -1
    _, end = min(endings)
    start = (a["cx"], a["cy"], 0)
    if end == start:
        return None
    while prev[end][0] != start:
        end = prev[end][0]
    return prev[end][1]


def act(obs, available, memory):
    obs = np.asarray(obs)
    allowed = {x.name if hasattr(x, "name") else str(x).upper()
               for x in available}
    geom = memory.get("geom")
    if geom is None:
        geom = geometry(obs)
        memory["geom"] = geom
    data = read_frame(obs, geom)
    old, last = memory.get("state"), memory.get("last")
    state, changed = update_state(data, old, last)
    changed |= learn_modes(state, data)

    def emit(name, row=None, col=None, record=None):
        if name not in allowed:
            name = next((x for x in ("ACTION5", "ACTION1", "ACTION6", "RESET")
                         if x in allowed), next(iter(allowed), "ACTION1"))
            if name != "ACTION6":
                row = col = None
            elif row is None:
                row = col = 0
        memory["last"] = record
        memory["state"] = state
        return name, row, col

    force_select = None
    if old is not None and last and last.get("kind") == "move":
        who = last.get("who")
        dx, dy = DIRS[last["action"]]
        if who and who[0] == "a" and who[1] < len(old["axes"]):
            i = who[1]
            if i >= len(state["axes"]):
                changed = True
            else:
                a, b = old["axes"][i], state["axes"][i]
                expected = a["p"] + (dx if a["d"] == "v" else dy)
                if b["p"] != expected:
                    changed = True
                    if b["p"] == a["p"] and expected != a["p"]:
                        force_select = who
        elif who and who[0] == "s" and who[1] < len(old["actors"]):
            i = who[1]
            if i < len(state["actors"]):
                a, b = old["actors"][i], state["actors"][i]
                if (a["cx"], a["cy"], a["pts"]) == (
                        b["cx"], b["cy"], b["pts"]):
                    force_select = who
                    changed = True

    if changed:
        memory.pop("plan", None)

    def select(who):
        p = click_point(who, state, data)
        row, col = display_point(p, geom)
        failures = memory.get("select_failures", 0)
        if last and last.get("kind") == "select" and last.get("who") == who:
            failures += 1
        else:
            failures = 0
        memory["select_failures"] = failures
        if failures >= 2 and "ACTION5" in allowed:
            return emit("ACTION5", record={"kind": "cycle"})
        return emit("ACTION6", row, col, {"kind": "select", "who": who})

    if force_select is not None:
        return select(force_select)

    # A directional action checks victory; clicks and selection cycles do not.
    if state["T"] and state["T"] <= data["O"] | data["G"]:
        for i, a in enumerate(state["axes"]):
            if not a["fixed"]:
                who = ("a", i)
                if state["selected"] != who:
                    return select(who)
                action = 1 if a["d"] == "v" else 3
                return emit("ACTION%d" % action,
                            record={"kind": "move", "who": who, "action": action})

    plan = memory.get("plan")
    if plan is None:
        plan = make_plan(state)
        memory["plan"] = plan

    if plan is not None:
        for i, (a, dest) in enumerate(zip(state["axes"], plan["axes"])):
            if a["p"] == dest:
                continue
            who = ("a", i)
            if state["selected"] != who:
                return select(who)
            action = (4 if dest > a["p"] else 3) if a["d"] == "v" else (
                2 if dest > a["p"] else 1)
            return emit("ACTION%d" % action,
                        record={"kind": "move", "who": who, "action": action})

        for i, a in enumerate(state["actors"]):
            if a["fixed"] or i not in plan["goals"]:
                continue
            move = first_move(a, plan["goals"][i], state["W"], state["H"])
            if move is None:
                continue
            if move == -1:
                memory.pop("plan", None)
                break
            who = ("s", i)
            if state["selected"] != who:
                return select(who)
            return emit("ACTION%d" % move,
                        record={"kind": "move", "who": who, "action": move})
        else:
            memory.pop("plan", None)

    # Informative fallback when a new mechanic prevents a complete modelled plan.
    movable = [(i, a) for i, a in enumerate(state["actors"]) if not a["fixed"]]
    if movable:
        selected = state["selected"]
        i, a = next(((i, a) for i, a in movable if selected == ("s", i)),
                    movable[0])
        who = ("s", i)
        if selected != who:
            return select(who)
        if state["T"]:
            tx = sum(x for x, y in state["T"]) / len(state["T"])
            ty = sum(y for x, y in state["T"]) / len(state["T"])
        else:
            tx, ty = state["W"] / 2, state["H"] / 2
        choices = []
        for action, (dx, dy) in DIRS.items():
            nx, ny = a["cx"] + dx, a["cy"] + dy
            x0, y0 = nx - a["w"] // 2, ny - a["h"] // 2
            if (0 <= x0 and 0 <= y0 and x0 + a["w"] <= state["W"]
                    and y0 + a["h"] <= state["H"]):
                choices.append((abs(nx - tx) + abs(ny - ty), action))
        if choices:
            _, action = min(choices)
            return emit("ACTION%d" % action,
                        record={"kind": "move", "who": who, "action": action})

    for i, a in enumerate(state["axes"]):
        if not a["fixed"]:
            who = ("a", i)
            if state["selected"] != who:
                return select(who)
            limit = state["W"] if a["d"] == "v" else state["H"]
            positive = a["p"] < limit // 2
            action = (4 if positive else 3) if a["d"] == "v" else (
                2 if positive else 1)
            return emit("ACTION%d" % action,
                        record={"kind": "move", "who": who, "action": action})

    return emit("ACTION5", record={"kind": "cycle"})

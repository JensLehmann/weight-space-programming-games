import numpy as np
import collections
import itertools
import time


# Geometry: (kind, width, height, horizontal_row, vertical_column,
#            naturally_solid_X_center)
# State: (x, y, geometry, color)

def _inside(g, u, v, selected=False):
    kind, w, h, r, c, special = g
    if not (0 <= u < w and 0 <= v < h):
        return False
    mx, my = w // 2, h // 2
    if (u, v) == (mx, my):
        if selected:
            return True
        if kind == "C":
            return r == my or c == mx
        if kind == "X":
            return special
        return False
    if kind == "C":
        return u == c or v == r
    if kind == "X":
        return u == v or u + v == w - 1
    if kind == "D":
        return abs(u - mx) + abs(v - my) == mx
    if kind == "R":
        return u in (0, w - 1) or v in (0, h - 1)
    return False


def _points(g, mem, selected=False):
    cache = mem.setdefault("points", {})
    key = (g, selected)
    if key in cache:
        return cache[key]
    kind, w, h, r, c, special = g
    p = set()
    if kind == "C":
        p.update((c, v) for v in range(h))
        p.update((u, r) for u in range(w))
    elif kind == "X":
        p.update((i, i) for i in range(w))
        p.update((i, w - 1 - i) for i in range(w))
    elif kind == "D":
        m = w // 2
        for v in range(h):
            d = m - abs(v - m)
            p.add((m - d, v))
            p.add((m + d, v))
    elif kind == "R":
        p.update((u, 0) for u in range(w))
        p.update((u, h - 1) for u in range(w))
        p.update((0, v) for v in range(h))
        p.update((w - 1, v) for v in range(h))
    center = (w // 2, h // 2)
    if _inside(g, *center, selected):
        p.add(center)
    else:
        p.discard(center)
    result = tuple(p)
    cache[key] = result
    return result


def _center(s):
    return s[0] + s[2][1] // 2, s[1] + s[2][2] // 2


def _zero(obs):
    yy, xx = np.where(obs[:63] == 0)
    return (int(xx[0]), int(yy[0])) if len(xx) else None


def _covered(s, p, selected=False):
    return _inside(s[2], p[0] - s[0], p[1] - s[1], selected)


def _fit_shape(obs, center, prior=None):
    cx, cy = center
    vals, counts = np.unique(obs[:63], return_counts=True)
    colors = [int(v) for v, n in zip(vals, counts)
              if v not in (0, 1, 2, 4, 5, 15) and n >= 3]
    best = (-1e30, None)

    def evaluate(kind, w, h, r, c, color):
        nonlocal best
        x, y = cx - w // 2, cy - h // 2
        if kind == "C":
            p = set((c, v) for v in range(h))
            p.update((u, r) for u in range(w))
        elif kind == "X":
            p = set((i, i) for i in range(w))
            p.update((i, w - 1 - i) for i in range(w))
        elif kind == "D":
            m = w // 2
            p = set()
            for v in range(h):
                d = m - abs(v - m)
                p.add((m - d, v))
                p.add((m + d, v))
        else:
            p = set((u, 0) for u in range(w))
            p.update((u, h - 1) for u in range(w))
            p.update((0, v) for v in range(h))
            p.update((w - 1, v) for v in range(h))
        p.discard((w // 2, h // 2))
        if not p:
            return
        a = np.asarray(tuple(p), dtype=int)
        xx, yy = a[:, 0] + x, a[:, 1] + y
        valid = (xx >= 0) & (xx < 64) & (yy >= 0) & (yy < 63)
        if not valid.any():
            return
        q = obs[yy[valid], xx[valid]]
        hit = int(np.count_nonzero(q == color))
        missing = int(np.count_nonzero((q == 5) | (q == 4)))
        hard = int(np.count_nonzero((q == 1) | (q == 2)))
        other = len(q) - hit - missing - hard
        score = hit - 5 * missing - 2 * hard - .4 * other
        score -= .3 * int(np.count_nonzero(~valid))
        if prior is not None:
            pk, pw, ph, pr, pc, ps = prior[2]
            if kind == pk and w == pw and h == ph:
                score += .15
        if hit >= 3 and score > best[0]:
            special = (prior is not None and kind == "X" and
                       prior[2][0] == "X" and prior[2][5])
            best = score, (x, y, (kind, w, h, r, c, special), color)

    for color in colors:
        for n in range(3, 64, 2):
            evaluate("C", n, 1, 0, n // 2, color)
            evaluate("C", 1, n, n // 2, 0, color)
        for n in range(5, 54, 2):
            evaluate("X", n, n, 0, 0, color)
            evaluate("D", n, n, 0, 0, color)
        for h in range(4, 50):
            for w in range(4, 50):
                if w + h <= 78:
                    evaluate("R", w, h, 0, 0, color)

        horizontal, vertical = [], []
        for shift in range(-27, 28, 3):
            ry, rx = cy + shift, cx + shift
            for n in range(5, 54, 2):
                half = n // 2
                if 0 <= ry < 63:
                    q = obs[ry, max(0, cx-half):min(64, cx+half+1)]
                    hit = int(np.count_nonzero(q == color))
                    miss = int(np.count_nonzero(
                        (q == 5) | (q == 4) | (q == 1) | (q == 2)))
                    if hit >= max(4, .6 * len(q)):
                        horizontal.append((hit - 5*miss, n, shift))
                if 0 <= rx < 64:
                    q = obs[max(0, cy-half):min(63, cy+half+1), rx]
                    hit = int(np.count_nonzero(q == color))
                    miss = int(np.count_nonzero(
                        (q == 5) | (q == 4) | (q == 1) | (q == 2)))
                    if hit >= max(4, .6 * len(q)):
                        vertical.append((hit - 5*miss, n, shift))
        horizontal.sort(reverse=True)
        vertical.sort(reverse=True)
        for _, w, dr in horizontal[:14]:
            for _, h, dc in vertical[:14]:
                r, c = h // 2 + dr, w // 2 + dc
                if 0 <= r < h and 0 <= c < w:
                    evaluate("C", w, h, r, c, color)
        if prior is not None and prior[2][0] == "C":
            _, w, h, r, c, special = prior[2]
            for rr in range(r % 3, h, 3):
                for cc in range(c % 3, w, 3):
                    evaluate("C", w, h, rr, cc, color)
    return best[1]


def _components(mask):
    seen, result = set(), []
    h, w = mask.shape
    for yy, xx in zip(*np.where(mask)):
        p = int(xx), int(yy)
        if p in seen:
            continue
        seen.add(p)
        todo, comp = [p], []
        while todo:
            x, y = todo.pop()
            comp.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = x + dx, y + dy
                if (0 <= q[0] < w and 0 <= q[1] < h and
                        mask[q[1], q[0]] and q not in seen):
                    seen.add(q)
                    todo.append(q)
        result.append(comp)
    return result


def _static_scene(obs, mem):
    if "paints" in mem:
        return
    paints = []
    for comp in _components(obs[:63] == 2):
        if len(comp) < 8:
            continue
        xs, ys = zip(*comp)
        x, y = min(xs), min(ys)
        w, h = max(xs)-x+1, max(ys)-y+1
        if not (4 <= w <= 10 and 4 <= h <= 10):
            continue
        q = obs[y+1:y+h-1, x+1:x+w-1]
        values, counts = np.unique(q, return_counts=True)
        choices = [(int(v), int(n)) for v, n in zip(values, counts)
                   if v not in (0, 1, 2, 4, 5, 15)]
        if choices:
            color = max(choices, key=lambda t: t[1])[0]
            paints.append((x, y, w, h, color))
    mem["paints"] = paints

    blocks = []
    for comp in _components(obs[:63] == 1):
        if len(comp) < 4:
            continue
        xs, ys = zip(*comp)
        x, y = min(xs), min(ys)
        w, h = max(xs)-x+1, max(ys)-y+1
        if (w, h) == (8, 8):
            rows = ("11111111", "11100111", "11000011", "10000001",
                    "10000001", "11000011", "11100111", "11111111")
            comp = [(x+u, y+v) for v in range(8) for u in range(8)
                    if rows[v][u] == "1"]
        blocks.append((x, y, w, h, tuple(comp)))
    mem["blocks"] = blocks


def _find_markers(obs, minimum):
    ring = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if dx or dy]
    candidates = []
    for y in range(1, 62):
        for x in range(1, 63):
            if obs[y, x] in (1, 2, 4, 5, 15):
                continue
            q = [int(obs[y+dy, x+dx]) for dx, dy in ring]
            score = q.count(4)
            if score >= minimum and 5 not in q:
                candidates.append((score, x, y))
    candidates.sort(reverse=True)
    result = []
    for score, x, y in candidates:
        if not any(abs(x-a) <= 2 and abs(y-b) <= 2 for a, b in result):
            result.append((x, y))
    return result


def _learn_targets(obs, mem):
    targets = mem.setdefault("targets", {})
    scan = mem.get("phase") == "scan"
    changed = False
    for p in _find_markers(obs, 3 if scan else 6):
        if p not in targets and not any(
                abs(p[0]-q[0]) <= 2 and abs(p[1]-q[1]) <= 2
                for q in targets):
            targets[p] = None
            changed = True
    if scan:
        return changed
    active, shapes = mem["active"], mem["shapes"]
    for p in targets:
        if targets[p] is None and not any(
                _covered(s, p, i == active) for i, s in enumerate(shapes)):
            color = int(obs[p[1], p[0]])
            if color not in (0, 1, 2, 4, 5, 15):
                targets[p] = color
                changed = True
    return changed


def _rect_hit(x, y, g, box, selected=False):
    bx, by, bw, bh = box[:4]
    w, h = g[1:3]
    left, right = max(x, bx), min(x+w, bx+bw)
    top, bottom = max(y, by), min(y+h, by+bh)
    if left >= right or top >= bottom:
        return False
    return any(_inside(g, u-x, v-y, selected)
               for v in range(top, bottom) for u in range(left, right))


def _block_hit(x, y, g, block):
    bx, by, bw, bh, points = block
    if x >= bx+bw or x+g[1] <= bx or y >= by+bh or y+g[2] <= by:
        return False
    return any(_inside(g, u-x, v-y, True) for u, v in points)


def _pose_step(x, y, g, action, mem):
    cache = mem.setdefault("transitions", {})
    key = x, y, g, action
    if key in cache:
        return cache[key]
    dx, dy = ((0, -3), (0, 3), (-3, 0), (3, 0))[action-1]
    nx, ny = x+dx, y+dy
    w, h = g[1:3]
    if not (0 <= nx+w//2 < 64 and 0 <= ny+h//2 < 64):
        result = x, y, g, ()
        cache[key] = result
        return result
    ng, early = g, False
    for block in mem["blocks"]:
        if not _block_hit(nx, ny, ng, block):
            continue
        kind, w, h, r, c, special = ng
        bx, by, bw, bh = block[:4]
        if kind in ("D", "R"):
            if dx:
                if w > 6:
                    nh, nw = h+3, w-3
                    ny += h//2 - nh//2
                    ny = round(ny/3)*3
                    if dx < 0:
                        nx -= dx
                    ng = "R", nw, nh, 0, 0, False
                else:
                    nx, ny = nx-dx, ny-dy
            else:
                if h > 6:
                    nh, nw = h-3, w+3
                    nx += w//2 - nw//2
                    nx = round(nx/3)*3
                    if dy < 0:
                        ny -= dy
                    ng = "R", nw, nh, 0, 0, False
                else:
                    nx, ny = nx-dx, ny-dy
            early = True
            break
        if kind != "C":
            nx, ny = x, y
            break
        vin = bx <= nx+c < bx+bw
        hin = by <= ny+r < by+bh
        if dx:
            sign = 1 if dx > 0 else -1
            if vin and hin:
                nx, ny = nx-dx, ny-dy
            elif vin:
                allowed = c > 0 if sign > 0 else c < w-2
                if allowed:
                    c = (c-3*sign) % w
                else:
                    nx, ny = nx-dx, ny-dy
            elif hin:
                nx, ny = nx-dx, ny-dy
                allowed = c < w-2 if sign > 0 else c > 0
                if allowed:
                    c = (c+3*sign) % w
        else:
            sign = 1 if dy > 0 else -1
            if vin and hin:
                nx, ny = nx-dx, ny-dy
            elif hin:
                allowed = r > 0 if sign > 0 else r < h-2
                if allowed:
                    r = (r-3*sign) % h
                else:
                    nx, ny = nx-dx, ny-dy
            elif vin:
                nx, ny = nx-dx, ny-dy
                allowed = r < h-2 if sign > 0 else r > 0
                if allowed:
                    r = (r+3*sign) % h
        ng = "C", w, h, r, c, special
    hits = ()
    if not early:
        hits = tuple(p[4] for p in mem["paints"]
                     if _rect_hit(nx, ny, ng, p))
    result = int(nx), int(ny), ng, hits
    cache[key] = result
    return result


def _step_state(state, action, mem):
    x, y, g, color = state
    nx, ny, ng, hits = _pose_step(x, y, g, action, mem)
    for pc in hits:
        if pc != color:
            color = pc
            break
    return nx, ny, ng, color


def _masks(state, targets):
    x, y, g, color = state
    good = bad = 0
    for i, (tx, ty, wanted) in enumerate(targets):
        if _inside(g, tx-x, ty-y):
            if color == wanted:
                good |= 1 << i
            else:
                bad |= 1 << i
    return good, bad


def _new_job(state):
    return {
        "queue": collections.deque([state]),
        "parents": {state: (None, 0, 0)},
        "candidates": {},
        "done": False,
        "expanded": 0
    }


def _search_job(job, mem, targets, deadline):
    queue, parents, candidates = (
        job["queue"], job["parents"], job["candidates"])
    count = 0
    while queue:
        if count % 128 == 0 and time.perf_counter() >= deadline:
            return
        s = queue.popleft()
        distance = parents[s][2]
        good, bad = _masks(s, targets)
        key = good, bad
        if (good or not bad) and key not in candidates:
            candidates[key] = distance, s
        count += 1
        job["expanded"] += 1
        if job["expanded"] >= 260000:
            queue.clear()
            break
        for a in (1, 2, 3, 4):
            ns = _step_state(s, a, mem)
            if ns not in parents:
                parents[ns] = s, a, distance+1
                queue.append(ns)
    job["done"] = True


def _path(job, state):
    result, parents = [], job["parents"]
    while parents[state][0] is not None:
        prev, a, distance = parents[state]
        result.append(a)
        state = prev
    return result[::-1]


def _combine(jobs, nt, order, clean):
    full = (1 << nt)-1
    dp = {0: (0, {})}
    for i in order:
        choices = [(good, bad, cost, state)
                   for (good, bad), (cost, state)
                   in jobs[i]["candidates"].items()
                   if not clean or bad == 0]
        ndp = {}
        for old, (cost, selected) in dp.items():
            for good, bad, extra, state in choices:
                mask = (old & ~bad) | good
                nc = cost+extra
                if mask not in ndp or nc < ndp[mask][0]:
                    pick = selected.copy()
                    pick[i] = state
                    ndp[mask] = nc, pick
        if len(ndp) > 12000:
            keep = sorted(ndp, key=lambda k:
                          k.bit_count()*100-ndp[k][0],
                          reverse=True)[:12000]
            ndp = {k: ndp[k] for k in keep}
        dp = ndp
        if not dp:
            return None
    return dp.get(full)


def _render_edges(mem):
    shapes, obs = mem["shapes"], mem.get("last_scan")
    edges = set()
    if obs is None:
        return edges
    for i in range(len(shapes)):
        for j in range(i+1, len(shapes)):
            a, b = shapes[i], shapes[j]
            if a[3] == b[3]:
                continue
            va = vb = 0
            for u, v in _points(a[2], mem):
                x, y = a[0]+u, a[1]+v
                if 0 <= x < 64 and 0 <= y < 63 and _covered(b, (x, y)):
                    color = int(obs[y, x])
                    va += color == a[3]
                    vb += color == b[3]
            if va > vb and va:
                edges.add((j, i))
            elif vb > va and vb:
                edges.add((i, j))
    return edges


def _make_plan(mem, targets):
    jobs = mem["jobs"]
    n = len(jobs)
    result = _combine(jobs, len(targets), range(n), True)
    if result is None and n <= 4:
        edges = _render_edges(mem)
        for order in itertools.permutations(range(n)):
            ranks = {i: k for k, i in enumerate(order)}
            if any(ranks[a] > ranks[b] for a, b in edges):
                continue
            candidate = _combine(jobs, len(targets), order, False)
            if candidate is not None and (
                    result is None or candidate[0] < result[0]):
                result = candidate
    if result is None:
        return None
    cost, chosen = result
    current, plan = mem["active"], []
    finals = list(mem["shapes"])
    for offset in range(n):
        i = (current+offset) % n
        finals[i] = chosen[i]
        plan.extend(_path(jobs[i], chosen[i]))
        if offset+1 < n:
            plan.append(5)
    ending = (current+n-1) % n
    positions = {(x, y) for x, y, color in targets}
    for k in range(n):
        i = (ending+k) % n
        if _center(finals[i]) not in positions:
            plan.extend([5]*k)
            break
    return plan or [5]


def _clear_search(mem):
    for key in ("jobs", "job_index", "goal_list"):
        mem.pop(key, None)


def _emit(a, mem, available):
    if mem.get("phase") != "scan" and mem.get("shapes"):
        if a == 5:
            mem["active"] = (mem["active"]+1) % len(mem["shapes"])
        elif 1 <= a <= 4:
            i = mem["active"]
            mem["shapes"][i] = _step_state(mem["shapes"][i], a, mem)
    name = "ACTION"+str(a)
    if name not in available:
        name = next((v for v in available if v != "RESET"), "RESET")
    return name, None, None


def _take_plan(mem, available):
    a = mem["plan"].pop(0)
    if not mem["plan"]:
        if mem.get("plan_full"):
            mem["finished_plan"] = True
        mem.pop("plan_full", None)
    return _emit(a, mem, available)


def _reveal_plan(mem):
    unknown = [p for p, color in mem["targets"].items() if color is None]
    active, n = mem["active"], len(mem["shapes"])
    for offset in range(n):
        i = (active+offset) % n
        s = mem["shapes"][i]
        relevant = [p for p in unknown if _covered(s, p, True)]
        if not relevant:
            continue
        if offset:
            return [5]*offset
        queue = collections.deque([s])
        parents = {s: (None, 0)}
        finish = None
        while queue and len(parents) < 16000:
            state = queue.popleft()
            if not any(_covered(state, p, True) for p in relevant):
                finish = state
                break
            for a in (1, 2, 3, 4):
                ns = _step_state(state, a, mem)
                if ns not in parents:
                    parents[ns] = state, a
                    queue.append(ns)
        if finish is not None:
            result = []
            while parents[finish][0] is not None:
                finish, a = parents[finish]
                result.append(a)
            return result[::-1]
        return [4, 2, 2]
    return [5]


def _validate_current(obs, mem):
    i = mem["active"]
    s = mem["shapes"][i]
    center = _zero(obs)
    if center is None:
        return False

    missing = 0
    for u, v in _points(s[2], mem, True):
        x, y = s[0]+u, s[1]+v
        if 0 <= x < 64 and 0 <= y < 63:
            missing += int(obs[y, x]) in (4, 5)

    changed = False
    if center != _center(s) or missing >= 2:
        fitted = _fit_shape(obs, center, s)
        if fitted is not None and fitted != s:
            mem["shapes"][i] = fitted
            s = fitted
            changed = True

    votes = collections.Counter()
    for u, v in _points(s[2], mem):
        x, y = s[0]+u, s[1]+v
        if 0 <= x < 64 and 0 <= y < 63:
            color = int(obs[y, x])
            if color not in (0, 1, 2, 4, 5, 15):
                votes[color] += 1
    if votes:
        color, count = votes.most_common(1)[0]
        if color != s[3] and count > .7*sum(votes.values()):
            mem["shapes"][i] = s[0], s[1], s[2], color
            changed = True
    return changed


def _begin_scan(mem):
    _clear_search(mem)
    mem.pop("plan", None)
    mem.pop("plan_full", None)
    mem.pop("finished_plan", None)
    mem["phase"] = "scan"
    mem["scan_centers"] = []
    mem["scan_frames"] = []
    mem["scan_after"] = {}
    mem["old_shapes"] = mem.get("shapes", [])
    mem["shapes"] = []
    mem.pop("previous_scan", None)
    mem["scan_wait"] = 0


def _finish_scan(obs, mem, active):
    shapes = mem["shapes"]
    frames = mem["scan_frames"]
    centers = mem["scan_centers"]
    after = mem["scan_after"]
    for i, s in enumerate(shapes):
        if s[2][0] != "X":
            continue
        p = centers[i]
        hidden = any(_covered(t, p) for j, t in enumerate(shapes) if j != i)
        frame = after.get(i)
        solid = (frame is not None and not hidden and
                 int(frame[p[1], p[0]]) == s[3])
        g = s[2][:-1]+(bool(solid),)
        shapes[i] = s[0], s[1], g, s[3]

    mem["active"] = active
    mem["phase"] = "play"
    mem["last_scan"] = obs.copy()
    for frame_index, frame in enumerate(frames):
        for p in mem["targets"]:
            if any(_covered(s, p, j == frame_index)
                   for j, s in enumerate(shapes)):
                continue
            color = int(frame[p[1], p[0]])
            if color not in (0, 1, 2, 4, 5, 15):
                mem["targets"][p] = color
    for key in ("scan_frames", "scan_after", "old_shapes", "previous_scan"):
        mem.pop(key, None)


def act(obs, available, memory):
    obs = np.asarray(obs, dtype=int)
    mem = memory
    started = time.perf_counter()

    # RESET is an engine action even when it is omitted from the ordinary
    # action list. First make a selection so RESET is guaranteed to reset
    # this level rather than restart the whole game.
    if "_boot" not in mem:
        mem["_boot"] = 1
        return "ACTION5", None, None
    if mem["_boot"] == 1:
        mem.clear()
        mem["_boot"] = 2
        return "RESET", None, None

    if "phase" not in mem:
        mem["targets"] = {}
        _static_scene(obs, mem)
        _begin_scan(mem)

    if mem.get("finished_plan"):
        mem.pop("finished_plan", None)
        mem["rescans"] = mem.get("rescans", 0)+1
        _begin_scan(mem)

    if mem["phase"] == "scan":
        _learn_targets(obs, mem)
        centers = mem["scan_centers"]
        shapes = mem["shapes"]
        center = _zero(obs)
        previous = mem.get("previous_scan")
        if previous is not None:
            mem["scan_after"][previous] = obs.copy()

        if center is not None and center in centers:
            _finish_scan(obs, mem, centers.index(center))
        elif center is not None:
            prior = next((s for s in mem.get("old_shapes", [])
                          if _center(s) == center), None)
            fitted = _fit_shape(obs, center, prior)
            if fitted is not None:
                centers.append(center)
                shapes.append(fitted)
                mem["scan_frames"].append(obs.copy())
                mem["previous_scan"] = len(shapes)-1
            if len(centers) >= 8:
                _finish_scan(obs, mem, 0)
            else:
                return _emit(5, mem, available)
        else:
            mem["scan_wait"] += 1
            if mem["scan_wait"] <= 8:
                return _emit(5, mem, available)
            # No reliable selected center: try a movement, then rescan.
            _begin_scan(mem)
            return "ACTION4", None, None

    if not mem.get("shapes"):
        return _emit(5, mem, available)

    changed = _validate_current(obs, mem)
    changed |= _learn_targets(obs, mem)
    if changed:
        mem.pop("plan", None)
        mem.pop("plan_full", None)
        _clear_search(mem)

    if mem.get("plan"):
        return _take_plan(mem, available)
    mem.pop("plan", None)

    if any(color is None for color in mem["targets"].values()):
        mem["plan"] = _reveal_plan(mem)
        return _take_plan(mem, available)

    targets = [(x, y, color) for (x, y), color in
               sorted(mem["targets"].items(), key=lambda z: (z[0][1], z[0][0]))]
    if not targets:
        return _emit(5, mem, available)

    if "jobs" not in mem:
        mem["jobs"] = [_new_job(s) for s in mem["shapes"]]
        mem["job_index"] = 0

    jobs = mem["jobs"]
    deadline = started+7.3
    k = mem["job_index"]
    while k < len(jobs):
        _search_job(jobs[k], mem, targets, deadline)
        if not jobs[k]["done"]:
            mem["job_index"] = k
            return _emit(5, mem, available)
        k += 1
        mem["job_index"] = k
        if time.perf_counter() >= deadline and k < len(jobs):
            return _emit(5, mem, available)

    plan = _make_plan(mem, targets)
    if plan is not None:
        mem["plan"] = plan
        mem["plan_full"] = True
        _clear_search(mem)
        return _take_plan(mem, available)

    full = (1 << len(targets))-1
    present = 0
    for s in mem["shapes"]:
        good, bad = _masks(s, targets)
        present |= good
    missing = full & ~present
    best = None
    for i, job in enumerate(jobs):
        for (good, bad), (cost, state) in job["candidates"].items():
            gain = (good & missing).bit_count()
            if gain and not bad:
                switches = (i-mem["active"]) % len(jobs)
                value = gain*100-cost-switches
                if best is None or value > best[0]:
                    best = value, i, state
    if best is not None:
        _, i, state = best
        switches = (i-mem["active"]) % len(jobs)
        plan = [5]*switches+_path(jobs[i], state)
        if plan:
            mem["plan"] = plan
            _clear_search(mem)
            return _take_plan(mem, available)

    _clear_search(mem)
    _begin_scan(mem)
    return _emit(5, mem, available)

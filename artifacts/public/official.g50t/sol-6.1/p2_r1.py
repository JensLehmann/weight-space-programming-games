import numpy as np
import heapq
import time
from collections import deque

D = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
PD = ((0, 1), (1, 0), (0, -1), (-1, 0))


def dst(p, dx, dy):
    x, y = p % 10 + dx, p // 10 + dy
    return y * 10 + x if 0 <= x < 10 and 0 <= y < 10 else -1


def md(a, b):
    return abs(a % 10 - b % 10) + abs(a // 10 - b // 10)


def patch(o, p, r):
    x, y = 4 + 6 * (p % 10), 4 + 6 * (p // 10)
    return o[y-r:y+r+1, x-r:x+r+1]


def actors(o, colour):
    a = np.full((5, 5), colour, dtype=np.int8)
    a[2, 2] = 5
    return [p for p in range(10, 100)
            if np.array_equal(patch(o, p, 2), a)]


def gate_pattern(colour, rotation):
    a = np.full((7, 7), 5, dtype=np.int8)
    a[1:6, 1:6] = colour
    a[6, 3] = colour
    if colour == 8:
        a[1, 2] = a[1, 4] = 5
    return np.rot90(a, -rotation)


def component_labels(o):
    labels = np.full((64, 64), -1, dtype=np.int16)
    n = 0
    for colour in (8, 11, 15):
        for yy, xx in zip(*np.where(o == colour)):
            x, y = int(xx), int(yy)
            if labels[y, x] >= 0:
                continue
            labels[y, x] = n
            q = deque([(x, y)])
            while q:
                x, y = q.popleft()
                for nx, ny in ((x-1, y), (x+1, y),
                               (x, y-1), (x, y+1)):
                    if (0 <= nx < 64 and 0 <= ny < 64
                            and labels[ny, nx] < 0
                            and o[ny, nx] == colour):
                        labels[ny, nx] = n
                        q.append((nx, ny))
            n += 1
    return labels


def update_heuristic(e):
    distances = [999] * 100
    distances[e["goal"]] = 0
    q = deque([e["goal"]])
    edges = {}
    for gr in e["groups"]:
        for a, b in gr["pairs"]:
            edges.setdefault(a, []).append(b)
            edges.setdefault(b, []).append(a)
    while q:
        p = q.popleft()
        for dx, dy in D.values():
            n = dst(p, dx, dy)
            if (n >= 0 and e["floor"][n]
                    and distances[n] > distances[p]+1):
                distances[n] = distances[p]+1
                q.append(n)
        for n in edges.get(p, []):
            if distances[n] > distances[p]:
                distances[n] = distances[p]
                q.appendleft(n)
    e["h"] = distances


def parse(o):
    ps = actors(o, 9)
    start = ps[0] if ps else 11
    labels = component_labels(o)

    # Terrain collision uses only the actor's centre. In particular,
    # three-pixel-wide passages must not be discarded.
    floor = [False] * 100
    ambiguous = set()
    for p in range(10, 100):
        z = patch(o, p, 2)
        if z.shape != (5, 5):
            continue
        centre = int(z[2, 2])
        count = np.count_nonzero(z)
        floor[p] = centre != 0 and (centre == 5 or count >= 9)
        if floor[p] and count < 24:
            ambiguous.add(p)

    gates, gate_nodes = [], set()
    for p in range(10, 100):
        z = patch(o, p, 3)
        if z.shape != (7, 7):
            continue
        found = False
        for colour in (8, 11):
            for r in range(4):
                pat = gate_pattern(colour, r)
                if np.array_equal(z, pat):
                    x, y = 4 + 6*(p % 10), 4 + 6*(p // 10)
                    gates.append({
                        "home": p, "open": dst(p, *PD[r]),
                        "colour": colour, "pattern": pat,
                        "component": int(labels[y, x]), "group": -1
                    })
                    gate_nodes.add(p)
                    found = True
                    break
            if found:
                break

    ports = []
    for p in range(10, 100):
        z = patch(o, p, 3)
        if (z.shape == (7, 7) and np.all(z[0] == 15)
                and np.all(z[-1] == 15)
                and np.all(z[:, 0] == 15)
                and np.all(z[:, -1] == 15)):
            x, y = 4 + 6*(p % 10), 4 + 6*(p // 10)
            ports.append((p, int(labels[y, x-3])))

    raw_buttons = []
    for p in range(10, 100):
        if p in gate_nodes:
            continue
        z = patch(o, p, 1)
        for colour in (8, 11, 15):
            if z.shape == (3, 3) and np.all(z == colour):
                x, y = 4 + 6*(p % 10), 4 + 6*(p // 10)
                raw_buttons.append({
                    "node": p, "colour": colour,
                    "component": int(labels[y, x]), "group": -1
                })
                break

    component_data = {}
    for g in gates:
        component_data.setdefault(g["component"], [])
    for p, c in ports:
        component_data.setdefault(c, []).append(p)

    groups, component_group = [], {}
    for c in component_data:
        if c >= 0:
            component_group[c] = len(groups)
            groups.append({"gates": [], "pairs": [], "buttons": []})

    for i, g in enumerate(gates):
        g["group"] = component_group.get(g["component"], -1)
        if g["group"] >= 0:
            groups[g["group"]]["gates"].append(i)

    for c, nodes in component_data.items():
        gi = component_group.get(c, -1)
        if gi < 0:
            continue
        remaining = list(nodes)
        while len(remaining) >= 2:
            _, a, b = min(
                (md(a, b), a, b)
                for i, a in enumerate(remaining)
                for b in remaining[i+1:]
            )
            groups[gi]["pairs"].append((a, b))
            remaining.remove(a)
            remaining.remove(b)

    buttons = []
    for b in raw_buttons:
        gi = component_group.get(b["component"], -1)
        if gi >= 0 and (groups[gi]["gates"] or groups[gi]["pairs"]):
            b["group"] = gi
            groups[gi]["buttons"].append(len(buttons))
            buttons.append(b)

    button_at = [-1] * 100
    for i, b in enumerate(buttons):
        button_at[b["node"]] = i

    goal = None
    for p in range(10, 100):
        x, y = 4 + 6*(p % 10), 4 + 6*(p // 10)
        if (o[y, x] == 9 and o[y-1, x] == 5 and o[y+1, x] == 5
                and o[y, x-1] == 5 and o[y, x+1] == 5):
            goal = p
            break
    if goal is None:
        goal = max((p for p in range(100) if floor[p]),
                   key=lambda p: md(start, p), default=start)

    slots = 0
    for k in range(8):
        z = o[1:4, 1+4*k:4+4*k]
        if z.shape != (3, 3):
            break
        ring = np.concatenate((z[0], z[2], z[1, :1], z[1, 2:]))
        if (np.all(z == 1)
                or (z[1, 1] == 0
                    and (np.all(ring == 9) or np.all(ring == 2)))):
            slots += 1
        else:
            break

    enemy_start = actors(o, 14)
    top = min((p // 10 for p in range(100) if floor[p]), default=1)
    cols = [p % 10 for p in range(100)
            if floor[p] and p // 10 == top]
    left, right = min(cols, default=0), max(cols, default=9)
    tracks = []
    for p in enemy_start:
        x, y = p % 10, p // 10
        end_x = x if y > top else right
        tr = {top*10+i for i in range(left, end_x+1)}
        tr.update(j*10+end_x for j in range(top, max(top, y)+1))
        tr.add(p)
        tracks.append(tr)

    e = {
        "start": start, "goal": goal, "floor": floor,
        "ambiguous": ambiguous,
        "gates": gates, "groups": groups, "buttons": buttons,
        "ba": button_at, "slots": max(2, slots),
        "es": enemy_start, "tracks": tracks, "cache": {}
    }
    update_heuristic(e)
    s = (start, (), tuple((p, 0, True) for p in enemy_start), 0, 0, 0)
    return e, s


def blocked(e, mask):
    c = e["cache"]
    if mask not in c:
        c[mask] = frozenset(
            g["open"] if mask & (1 << i) else g["home"]
            for i, g in enumerate(e["gates"])
        )
    return c[mask]


def signals(e, positions):
    result = 0
    for p in positions:
        if 0 <= p < 100 and e["ba"][p] >= 0:
            result |= 1 << e["buttons"][e["ba"][p]]["group"]
    return result


def step_model(e, state, action, scripts, cap):
    p, ghosts, enemies, doors, sig, tick = state
    target = dst(p, *D[action])
    if (target < 0 or not e["floor"][target]
            or target in blocked(e, doors)):
        return None

    ng = len(ghosts)
    old = [p] + list(ghosts) + [a[0] for a in enemies]
    future = old[:]
    enemy_data = [list(a) for a in enemies]
    occupancy = [0] * len(e["buttons"])
    for i, pos in enumerate(old):
        if 0 <= pos < 100 and e["ba"][pos] >= 0:
            occupancy[e["ba"][pos]] |= 1 << i
    closing = 0
    teleports = []

    def event(group, value):
        nonlocal doors, sig, closing
        if group < 0:
            return
        bit = 1 << group
        if bool(sig & bit) == value:
            return
        sig = sig | bit if value else sig & ~bit
        gr = e["groups"][group]
        for gi in gr["gates"]:
            gb = 1 << gi
            was = bool(doors & gb)
            if e["gates"][gi]["colour"] == 11:
                if not value:
                    continue
                now = not was
            else:
                now = value
            doors = doors | gb if now else doors & ~gb
            if was and not now:
                closing |= gb
        if value and gr["pairs"]:
            teleports.append(group)

    def leave(i):
        pos = old[i]
        if not 0 <= pos < 100:
            return
        bi = e["ba"][pos]
        if bi >= 0:
            before = occupancy[bi]
            occupancy[bi] &= ~(1 << i)
            if before and not occupancy[bi]:
                event(e["buttons"][bi]["group"], False)

    leave(0)
    future[0] = target
    for i, pos in enumerate(ghosts):
        if i < len(scripts) and tick < len(scripts[i]):
            n = dst(pos, *D[scripts[i][tick]])
            if n >= 0 and e["floor"][n] and n not in blocked(e, doors):
                leave(i+1)
                future[i+1] = n

    for k, a in enumerate(enemy_data):
        ep, direction, alive = a
        if not alive:
            continue
        for _ in range(4):
            choice = None
            for d in (direction, (direction-1) % 4,
                      (direction+1) % 4, (direction+2) % 4):
                n = dst(ep, *PD[d])
                if n in e["tracks"][k]:
                    choice = (n, d)
                    break
            if choice is None:
                choice = (ep, direction)
            n, direction = choice
            if n >= 0 and e["floor"][n] and n not in blocked(e, doors):
                leave(1+ng+k)
                future[1+ng+k] = n
                a[0] = n
                break
            direction = (direction+2) % 4
        a[1] = direction

    def safe():
        nonlocal closing
        for gi, g in enumerate(e["gates"]):
            bit = 1 << gi
            if closing & bit and not doors & bit:
                if future[0] == g["home"]:
                    return False
                for k, a in enumerate(enemy_data):
                    if a[2] and future[1+ng+k] == g["home"]:
                        a[2] = False
        closing = 0
        return not any(a[2] and future[1+ng+k] == future[0]
                       for k, a in enumerate(enemy_data))

    if not safe():
        return None
    order = [0] + list(range(1+ng, len(old))) + list(range(1, 1+ng))
    for i in order:
        pos = future[i]
        if 0 <= pos < 100:
            bi = e["ba"][pos]
            if bi >= 0:
                before = occupancy[bi]
                occupancy[bi] |= 1 << i
                if not before:
                    event(e["buttons"][bi]["group"], True)
    if not safe():
        return None

    for group in dict.fromkeys(teleports):
        for a, b in e["groups"][group]["pairs"]:
            for i, pos in enumerate(future):
                if pos == a:
                    future[i] = b
                elif pos == b:
                    future[i] = a
    for k, a in enumerate(enemy_data):
        a[0] = future[1+ng+k]
        if a[2] and a[0] == future[0]:
            return None

    return (future[0], tuple(future[1:1+ng]),
            tuple(tuple(a) for a in enemy_data),
            doors, sig, min(tick+1, cap))


def path_from(parents, s):
    result = []
    while parents[s] is not None:
        s, a = parents[s]
        result.append(a)
    return result[::-1]


def search(e, start, scripts, history, deadline, collect, recorded):
    cap = max([0] + [len(s) for s in scripts])
    start = start[:5] + (min(start[5], cap),)
    parents = {start: None}
    costs = {start: 0}
    serial = 0
    heap = [(min(100, e["h"][start[0]]), 0, serial, start)]
    candidates, counts = [], {}
    max_depth = max(8, min(85, 124-len(history)))
    expanded = 0

    while heap and len(parents) < 75000:
        _, depth, _, s = heapq.heappop(heap)
        if costs.get(s) != depth:
            continue
        if s[0] == e["goal"]:
            return path_from(parents, s), candidates

        if collect and depth+len(history) > 0:
            bi = e["ba"][s[0]]
            if bi >= 0:
                b = e["buttons"][bi]
                n = counts.get(bi, 0)
                if b["node"] not in recorded and n < 2:
                    counts[bi] = n+1
                    candidates.append((bi, path_from(parents, s), s))
        if depth >= max_depth:
            continue
        expanded += 1
        if expanded % 64 == 0 and time.perf_counter() >= deadline:
            break

        for a in D:
            new = step_model(e, s, a, scripts, cap)
            nd = depth+1
            if new is not None and nd < costs.get(new, 100000):
                costs[new] = nd
                parents[new] = (s, a)
                serial += 1
                h = min(100, e["h"][new[0]])
                heapq.heappush(heap, (nd+h, nd, serial, new))
    return None, candidates


def register(e, end, scripts, baseline):
    enemies = tuple(
        (p, end[2][k][1] if k < len(end[2]) else 0, True)
        for k, p in enumerate(e["es"])
    )
    ghosts = tuple(e["start"] for _ in scripts)
    mask = baseline
    for i, g in enumerate(e["gates"]):
        bit = 1 << i
        if g["colour"] == 11:
            if g["group"] >= 0 and end[4] & (1 << g["group"]):
                mask ^= bit
        else:
            mask &= ~bit
    sig = signals(e, [e["start"]] + list(ghosts)
                  + [a[0] for a in enemies])
    for i, g in enumerate(e["gates"]):
        if (g["colour"] != 11 and g["group"] >= 0
                and sig & (1 << g["group"])):
            mask |= 1 << i
    return (e["start"], ghosts, enemies, mask, sig, 0)


def make_plan(e, state, scripts, history, baseline, ends, deadline):
    serial = 0
    heap = [(0, serial, [], state, scripts, history,
             frozenset(ends), baseline)]
    seen = set()
    fallback = None
    iterations = 0

    while heap and time.perf_counter() < deadline and iterations < 32:
        _, _, prefix, s, programs, hist, recorded, base = heapq.heappop(heap)
        iterations += 1
        collect = len(programs) < e["slots"]-1
        solution, candidates = search(
            e, s, programs, hist, deadline, collect, recorded)
        if solution is not None:
            return prefix+solution
        if not collect:
            continue

        candidates.sort(key=lambda c: (
            e["buttons"][c[0]]["colour"] == 11, len(c[1]),
            md(e["buttons"][c[0]]["node"], e["goal"])))
        for bi, path, end in candidates:
            program = list(hist)+path
            if not program or len(program) > 80:
                continue
            node = e["buttons"][bi]["node"]
            new_programs = programs+[program]
            key = tuple(tuple(p) for p in new_programs)
            if key in seen:
                continue
            seen.add(key)
            new_prefix = prefix+path+[5]
            if fallback is None:
                fallback = new_prefix
            if len(new_prefix) > 118:
                continue
            ns = register(e, end, new_programs, base)
            serial += 1
            heapq.heappush(heap, (
                len(new_prefix)+2*len(new_programs),
                serial, new_prefix, ns, new_programs, [],
                recorded | {node}, ns[3]))
    return fallback or []


def read_doors(o, e, mask):
    for i, g in enumerate(e["gates"]):
        home = np.array_equal(patch(o, g["home"], 3), g["pattern"])
        opened = (g["open"] >= 10 and np.array_equal(
            patch(o, g["open"], 3), g["pattern"]))
        if opened and not home:
            mask |= 1 << i
        elif home and not opened:
            mask &= ~(1 << i)
    return mask


def observe(o, e, predicted, previous=None):
    p, ghosts, enemies, doors, sig, tick = predicted
    ps = actors(o, 9)
    if ps:
        p = min(ps, key=lambda q: md(q, p))

    ghosts = list(ghosts)
    unused = set(range(len(ghosts)))
    for pos in actors(o, 2):
        if not unused:
            break
        i = min(unused, key=lambda k: md(pos, ghosts[k]))
        ghosts[i] = pos
        unused.remove(i)

    enemies = [list(a) for a in enemies]
    unused = set(range(len(enemies)))
    for pos in actors(o, 14):
        if not unused:
            break
        k = min(unused, key=lambda i: md(pos, enemies[i][0]))
        unused.remove(k)
        if previous is not None:
            oldp = previous[2][k][0]
            for d, (dx, dy) in enumerate(PD):
                if dst(oldp, dx, dy) == pos:
                    enemies[k][1] = d
                    break
        enemies[k][0], enemies[k][2] = pos, True
        e["tracks"][k].add(pos)
    for k in unused:
        enemies[k][2] = False

    doors = read_doors(o, e, doors)
    sig = signals(e, [p]+ghosts+[a[0] for a in enemies])
    return (p, tuple(ghosts), tuple(tuple(a) for a in enemies),
            doors, sig, tick)


def spatial_path(e, start, target, doors, relaxed=False):
    blocks = blocked(e, doors)
    closed_homes = {
        g["home"] for i, g in enumerate(e["gates"])
        if not doors & (1 << i)
    }
    heap = [(0, start)]
    costs, parent = {start: 0}, {start: None}
    while heap:
        cost, p = heapq.heappop(heap)
        if cost != costs[p]:
            continue
        if p == target:
            path = []
            while parent[p] is not None:
                prev, action = parent[p]
                path.append((action, p))
                p = prev
            return path[::-1]
        for a, (dx, dy) in D.items():
            n = dst(p, dx, dy)
            if n < 0 or not e["floor"][n]:
                continue
            if n in blocks:
                if not relaxed or n not in closed_homes:
                    continue
                extra = 7
            else:
                extra = 0
            nc = cost+1+extra
            if nc < costs.get(n, 100000):
                costs[n] = nc
                parent[n] = (p, a)
                heapq.heappush(heap, (nc, n))
    return None


def reactive(e, state, scripts, memory):
    p, ghosts, enemies, doors, sig, tick = state
    blocks = blocked(e, doors)
    cap = max([0]+[len(s) for s in scripts])
    route = spatial_path(e, p, e["goal"], doors)
    target = e["goal"]

    if route is None:
        optimistic = spatial_path(e, p, e["goal"], doors, True)
        patrol_nodes = set().union(*e["tracks"]) if e["tracks"] else set()
        if optimistic:
            for _, node in optimistic:
                gis = [i for i, g in enumerate(e["gates"])
                       if g["home"] == node and not doors & (1 << i)
                       and g["colour"] == 11]
                choices = []
                for gi in gis:
                    gr = e["gates"][gi]["group"]
                    if gr < 0:
                        continue
                    for bi in e["groups"][gr]["buttons"]:
                        b = e["buttons"][bi]
                        if b["node"] in patrol_nodes:
                            continue
                        br = spatial_path(e, p, b["node"], doors, True)
                        if br is not None:
                            choices.append((len(br), b["node"]))
                if choices:
                    target = min(choices)[1]
                    break

        route = spatial_path(e, p, target, doors)
        if route is None:
            route = spatial_path(e, p, target, doors, True)

    if route:
        action, node = route[0]
        if node not in blocks:
            if step_model(e, state, action, scripts, cap) is not None:
                return action

    candidates = []
    visits = memory["visits"]
    for a in D:
        new = step_model(e, state, a, scripts, cap)
        if new is None:
            continue
        n = new[0]
        threat = min((md(n, en[0]) for en in new[2] if en[2]), default=20)
        remaining = spatial_path(e, n, target, new[3], True)
        route_cost = len(remaining) if remaining is not None else 100
        bi = e["ba"][n]
        button_penalty = (
            8 if bi >= 0 and e["buttons"][bi]["colour"] == 11 else 0)
        score = (route_cost + button_penalty
                 + 0.12*visits.get(n, 0)
                 + (8 if threat <= 1 else 0))
        candidates.append((score, a))

    if candidates:
        return min(candidates)[1]
    return 5 if memory["history"] else None


def act(obs, available, memory):
    o = np.asarray(obs, dtype=np.int8)
    deadline = time.perf_counter()+7.6

    # A single frame does not expose an existing recording's action history.
    if not memory.get("started"):
        memory["started"] = True
        return ("RESET", None, None)

    if "env" not in memory:
        e, s = parse(o)
        memory.update({
            "env": e, "state": s, "scripts": [], "history": [],
            "ends": [], "baseline": s[3], "plan": [],
            "visits": {}, "search_fail": 0,
            "image": o.copy()
        })

    e, s = memory["env"], memory["state"]
    scripts, history, ends = (
        memory["scripts"], memory["history"], memory["ends"])
    pending = memory.pop("pending", None)

    if pending is not None:
        old = s
        if pending == 5 and history:
            scripts = scripts+[list(history)]
            ends = ends+[old[0]]
            if len(scripts) >= e["slots"]:
                scripts, ends = [], []
            predicted = register(e, old, scripts, memory["baseline"])
            history = []
            s = observe(o, e, predicted)
            s = s[:5]+(0,)
            memory["baseline"] = s[3]
            memory["search_fail"] = 0
            if s[:5] != predicted[:5]:
                memory["plan"] = []

        elif pending in D:
            cap = max([0]+[len(x) for x in scripts])
            predicted = step_model(e, old, pending, scripts, cap)
            legal = predicted is not None
            if predicted is None:
                predicted = old
            actual = observe(o, e, predicted, old)

            unchanged = np.array_equal(
                o[7:62, :], memory["image"][7:62, :])
            rejected = unchanged and actual[0] == old[0]

            if rejected:
                legal = False
                actual = observe(o, e, old)
                n = dst(old[0], *D[pending])
                # Wires and gate sleeves can hide transparent terrain.
                # A rejected move provides a direct terrain measurement.
                if (n >= 0 and n not in blocked(e, old[3])
                        and e["floor"][n]):
                    e["floor"][n] = False
                    update_heuristic(e)
                    memory["search_fail"] = 0
                memory["plan"] = []
            elif actual[0] == old[0] and predicted[0] != old[0]:
                legal = False

            if legal:
                history = history+[pending]
            s = actual[:5]+(min(len(history), cap),)
            if actual[:4] != predicted[:4]:
                memory["plan"] = []
                memory["search_fail"] = 0
        else:
            s = observe(o, e, old, old)

    memory["scripts"] = scripts
    memory["history"] = history
    memory["ends"] = ends
    memory["state"] = s
    memory["image"] = o.copy()
    visits = memory["visits"]
    visits[s[0]] = visits.get(s[0], 0)+1

    plan_actions = memory.get("plan", [])
    retry = (memory["search_fail"] == 0
             or len(scripts) < e["slots"]-1
             or memory.get("last_doors") != s[3])
    if not plan_actions and retry:
        plan_actions = make_plan(
            e, s, scripts, history, memory["baseline"], ends, deadline)
        memory["plan"] = plan_actions
        memory["search_fail"] = 0 if plan_actions else 1
    memory["last_doors"] = s[3]

    action = plan_actions.pop(0) if plan_actions else None
    cap = max([0]+[len(x) for x in scripts])
    if action in D and step_model(e, s, action, scripts, cap) is None:
        memory["plan"] = []
        action = None
    if action == 5 and not history:
        memory["plan"] = []
        action = None

    if action is None:
        action = reactive(e, s, scripts, memory)
    if action is None:
        memory.pop("env", None)
        return ("RESET", None, None)

    name = "ACTION"+str(action)
    if available and name not in available:
        alternatives = [a for a in available
                        if a in ("ACTION1", "ACTION2", "ACTION3",
                                 "ACTION4", "ACTION5")]
        if alternatives:
            name = alternatives[0]
            action = int(name[6:])
    memory["pending"] = action
    return (name, None, None)

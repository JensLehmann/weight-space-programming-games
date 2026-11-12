import math
import time
import heapq
from collections import deque
import numpy as np


COLOURS = (12, 9, 14, 8)
SHAPES = (
    ((1, 1, 0), (0, 1, 1), (1, 0, 1)),
    ((0, 1, 0), (0, 1, 0), (1, 1, 1)),
    ((1, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (1, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (1, 1, 0), (0, 1, 1)),
    ((1, 1, 1), (0, 0, 1), (1, 0, 1)),
)
DIRECTIONS = ((0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4))
UNITS = (16, 4, 1)
INF = 10 ** 7

GLYPHS = {}
for s, shape in enumerate(SHAPES):
    a = np.array(shape, dtype=np.int8)
    for r in range(4):
        b = np.rot90(a, -r).ravel()
        mask = sum(int(v) << i for i, v in enumerate(b))
        GLYPHS[mask] = (s, r)

ADD = []
SUB = []
for d in range(96):
    ds, dc, dr = d // 16, (d // 4) % 4, d % 4
    ADD.append([
        ((c // 16 + ds) % 6) * 16
        + (((c // 4) % 4 + dc) % 4) * 4
        + ((c % 4 + dr) % 4)
        for c in range(96)
    ])
    SUB.append([
        ((c // 16 - ds) % 6) * 16
        + (((c // 4) % 4 - dc) % 4) * 4
        + ((c % 4 - dr) % 4)
        for c in range(96)
    ])


def decode_glyph(a):
    vals = [int(v) for v in np.asarray(a).ravel()]
    for ci, colour in enumerate(COLOURS):
        if colour not in vals:
            continue
        if any(v != 5 and v != colour for v in vals):
            continue
        mask = sum(int(v == colour) << i for i, v in enumerate(vals))
        sr = GLYPHS.get(mask)
        if sr is not None:
            return sr[0] * 16 + ci * 4 + sr[1]
    return None


def find_player(obs, old):
    for y, x in np.argwhere(obs[:60, :60] == 12):
        y, x = int(y), int(x)
        if x < 4 or x + 5 > 64 or y + 5 > 60:
            continue
        a = obs[y:y + 5, x:x + 5]
        if np.all(a[:2] == 12) and np.all(a[2:] == 9):
            return x, y
    return tuple(old) if old is not None else (34, 45)


def cell_distance(p, q, coords):
    x, y = coords[p]
    u, v = coords[q]
    return (abs(x - u) + abs(y - v)) // 5


def initial_cycle_index(cycle, p, coords):
    choices = [i for i, q in enumerate(cycle) if q == p]
    if not choices:
        return 0
    cells = set(cycle)
    x, y = coords[p]
    lookup = {tuple(xy): i for i, xy in enumerate(coords)}
    desired = None
    for dx, dy in ((0, 5), (-5, 0), (5, 0), (0, -5)):
        q = lookup.get((x + dx, y + dy))
        if q in cells:
            desired = q
            break
    for i in choices:
        if cycle[(i + 1) % len(cycle)] == desired:
            return i
    return choices[0]


def track_pads(m, sightings, moved, reset):
    pads, coords, tick = m["pads"], m["coords"], m["tick"]

    for pad in pads:
        pad["seen"] = False
        if reset:
            pad["hist"] = []
            pad["lt"] = -10000
        elif moved and pad.get("cycle"):
            pad["idx"] = (pad.get("idx", 0) + 1) % len(pad["cycle"])
            pad["p"] = pad["cycle"][pad["idx"]]

    unmatched = set(range(len(pads)))
    assignments = []
    pending = list(sightings)

    for typ, p in list(pending):
        exact = [
            i for i in unmatched
            if pads[i]["typ"] == typ and pads[i]["p"] == p
        ]
        if exact:
            i = exact[0]
            unmatched.remove(i)
            assignments.append((i, typ, p))
            pending.remove((typ, p))

    for typ, p in pending:
        candidates = [i for i in unmatched if pads[i]["typ"] == typ]
        i = None
        if candidates:
            candidates.sort(key=lambda j: cell_distance(pads[j]["p"], p, coords))
            j = candidates[0]
            dt = max(1, tick - pads[j].get("lt", tick - 1))
            if reset or cell_distance(pads[j]["p"], p, coords) <= dt:
                i = j
                unmatched.remove(i)
        if i is None:
            i = len(pads)
            pads.append({
                "typ": typ, "p": p, "last": p, "lt": tick,
                "hist": [], "dynamic": False, "cycle": [], "idx": 0,
            })
        assignments.append((i, typ, p))

    for i, typ, p in assignments:
        pad = pads[i]
        oldp, oldt = pad.get("last", p), pad.get("lt", -10000)
        if not reset and tick > oldt and oldp != p:
            pad["dynamic"] = True

        hist = pad.get("hist", [])
        if hist and hist[-1][0] == tick:
            hist[-1] = (tick, p)
        elif hist and hist[-1][0] == tick - 1:
            hist.append((tick, p))
        else:
            hist = [(tick, p)]
        hist = hist[-100:]
        pad["hist"] = hist

        cyc = pad.get("cycle", [])
        if cyc:
            choices = [j for j, q in enumerate(cyc) if q == p]
            if choices:
                if reset:
                    idx = initial_cycle_index(cyc, p, coords)
                elif cyc[pad.get("idx", 0)] == p:
                    idx = pad.get("idx", 0)
                else:
                    paired = [
                        j for j in choices
                        if oldt == tick - 1 and cyc[(j - 1) % len(cyc)] == oldp
                    ]
                    idx = paired[0] if paired else choices[0]
                pad["idx"] = idx
            else:
                pad["cycle"] = []
                cyc = []

        if not cyc and pad.get("dynamic") and len(hist) >= 4:
            ps = [q for _, q in hist]
            pair = (ps[-2], ps[-1])
            if pair[0] != pair[1]:
                for j in range(1, len(ps) - 2):
                    if (ps[j - 1], ps[j]) == pair:
                        cycle = ps[j:-1]
                        if len(cycle) >= 2:
                            pad["cycle"] = cycle
                            pad["idx"] = 0
                            break

        pad["p"] = p
        pad["last"] = p
        pad["lt"] = tick
        pad["seen"] = True


def read_frame(obs, m):
    old = m.get("prev")
    xy = find_player(obs, old["xy"] if old else None)

    if "coords" not in m:
        ox, oy = xy[0] % 5, xy[1] % 5
        coords = [(x, y) for y in range(oy, 60, 5)
                  for x in range(ox, 60, 5)]
        m.update({
            "coords": coords, "board": [-1] * len(coords),
            "goals": [], "refills": [], "pads": [], "pushers": {},
            "tick": 0, "decrement": 1, "reset_count": 0,
            "calls": 0, "plan": [],
        })

    coords = m["coords"]
    lookup = {tuple(q): i for i, q in enumerate(coords)}
    p = lookup.get(tuple(xy), 0)
    lives = sum(int(obs[61, 56 + 3 * i]) == 8 for i in range(3))

    bar = 0
    for x in range(13, 56):
        if int(obs[61, x]) in (3, 11):
            bar += 1
        else:
            break
    maximum = max(1, bar, m.get("maximum", 0))
    energy = sum(int(obs[61, x]) == 11
                 for x in range(13, min(56, 13 + maximum)))
    m["maximum"] = maximum

    reset = bool(old and (
        old["action"] == "RESET" or lives < old.get("lives", lives)
    ))
    moved = bool(old and tuple(old["xy"]) != tuple(xy) and not reset)
    if moved or reset:
        m["tick"] += 1
    if old and not reset:
        delta = old["energy"] - energy
        if delta in (1, 2):
            m["decrement"] = delta

    cfg = decode_glyph(obs[55:61:2, 3:9:2])
    if cfg is None:
        cfg = old["cfg"] if old else 0

    fog = int(np.count_nonzero(obs[:60, 4:64] == 5)) > 500
    m["fog"] = fog
    full = [False] * len(coords)
    sightings = []
    seen_goals = {}
    seen_refills = set()

    for i, (x, y) in enumerate(coords):
        if x + 5 > 64 or y + 5 > 64:
            continue
        if x < 14 and y >= 50:
            m["board"][i] = 1
            continue

        visible = not fog or all(
            (xx - xy[0] - 1.5) ** 2 + (yy - xy[1] - 1.5) ** 2 <= 400
            for xx, yy in ((x, y), (x + 4, y), (x, y + 4), (x + 4, y + 4))
        )
        full[i] = visible
        a = obs[y:y + 5, x:x + 5]
        centre_visible = (
            not fog or
            (x + 2 - xy[0] - 1.5) ** 2
            + (y + 2 - xy[1] - 1.5) ** 2 <= 400
        )
        if centre_visible:
            fours = int(np.count_nonzero(a == 4))
            if int(a[2, 2]) == 4 or (visible and fours >= 9):
                m["board"][i] = 1
            elif int(a[2, 2]) not in (4, 5) or (
                visible and np.count_nonzero(a == 3) >= 8
            ):
                m["board"][i] = 0

        if not visible:
            continue
        sub = a[1:4, 1:4]
        gc = decode_glyph(sub)
        if gc is not None and np.count_nonzero(a == 5) >= 17:
            seen_goals[i] = gc
            m["board"][i] = 0
            continue

        ring = [sub[0, 0], sub[0, 1], sub[0, 2],
                sub[1, 0], sub[1, 2],
                sub[2, 0], sub[2, 1], sub[2, 2]]
        if all(int(v) == 11 for v in ring):
            seen_refills.add(i)
            m["board"][i] = 0

        if i == p:
            continue
        z = int(np.count_nonzero(sub == 0))
        typ = None
        if z and any(np.any(sub == c) for c in COLOURS):
            typ = 1
        elif z and np.any(sub == 1):
            typ = 2
        elif z >= 2:
            typ = 0
        if typ is not None:
            sightings.append((typ, i))
            m["board"][i] = 0

    m["board"][p] = 0

    for gp, gc in seen_goals.items():
        found = next((g for g in m["goals"] if g["p"] == gp), None)
        if found is None:
            m["goals"].append({"p": gp, "cfg": gc, "done": False})
        else:
            found["cfg"] = gc
            found["done"] = False
    for g in m["goals"]:
        gp = g["p"]
        if full[gp] and gp not in seen_goals:
            x, y = coords[gp]
            a = obs[y:y + 5, x:x + 5]
            if gp == p or np.count_nonzero(a == 3) >= 12:
                g["done"] = True
        m["board"][gp] = 0

    for rp in sorted(seen_refills):
        if rp not in m["refills"]:
            m["refills"].append(rp)
    used = m.get("used", 0)
    for j, rp in enumerate(m["refills"]):
        if rp in seen_refills:
            used &= ~(1 << j)
        elif full[rp]:
            x, y = coords[rp]
            if not np.any(obs[y:y + 5, x:x + 5] == 11):
                used |= 1 << j
    m["used"] = used

    xs = sorted(set(x for x, _ in coords))
    ys = sorted(set(y for _, y in coords))
    for x in xs:
        for y in range(60):
            if x + 5 <= 64 and np.all(obs[y, x:x + 5] == 1):
                q = lookup.get((x, y + 1))
                direction = (0, 1)
                if q is None:
                    q = lookup.get((x, y - 5))
                    direction = (0, -1)
                if q is not None:
                    m["pushers"][q] = direction
                    m["board"][q] = 0
    for y in ys:
        for x in range(4, 64):
            if y + 5 <= 64 and np.all(obs[y:y + 5, x] == 1):
                q = lookup.get((x + 1, y))
                direction = (1, 0)
                if q is None:
                    q = lookup.get((x - 5, y))
                    direction = (-1, 0)
                if q is not None:
                    m["pushers"][q] = direction
                    m["board"][q] = 0

    track_pads(m, sightings, moved, reset)
    gm = sum(1 << i for i, g in enumerate(m["goals"]) if g["done"])
    m["calls"] += 1
    return {
        "p": p, "xy": tuple(xy), "cfg": cfg, "energy": energy,
        "lives": lives, "gm": gm, "rm": used, "reset": reset,
        "oldp": lookup.get(tuple(old["xy"])) if old else None,
    }


def make_model(m):
    coords = m["coords"]
    lookup = {tuple(q): i for i, q in enumerate(coords)}
    board = m["board"]
    walls = {i for i, v in enumerate(board) if v == 1}
    walls.update(g["p"] for g in m["goals"])

    edges = [[] for _ in coords]
    for p, (x, y) in enumerate(coords):
        if board[p] != 0:
            continue
        for dx, dy, act in DIRECTIONS:
            q = lookup.get((x + 5 * dx, y + 5 * dy))
            if q is None or board[q] != 0:
                continue
            dest, pushing = q, False
            if q in m["pushers"]:
                ux, uy = m["pushers"][q]
                qx, qy = coords[q]
                for k in range(1, 12):
                    z = lookup.get((qx + 5 * ux * k, qy + 5 * uy * k))
                    if z in walls:
                        if k > 1:
                            dest = lookup.get(
                                (qx + 5 * ux * (k - 1),
                                 qy + 5 * uy * (k - 1)), q
                            )
                            pushing = dest != q
                        break
            if board[dest] == 0:
                edges[p].append((act, q, dest, pushing))

    sequences = []
    period = 1
    options = [{0} for _ in coords]
    for pad in m["pads"]:
        typ = pad["typ"]
        unit = UNITS[typ]
        cyc = pad.get("cycle", [])
        if cyc:
            idx = pad.get("idx", 0)
            seq = cyc[idx:] + cyc[:idx]
            period = period * len(seq) // math.gcd(period, len(seq))
            cells = set(seq)
            dynamic = True
        else:
            seq = [pad["p"]]
            cells = {pad["p"]}
            dynamic = pad.get("dynamic", False)
            if dynamic:
                cells.update(p for _, p in pad.get("hist", []))
        sequences.append((typ, seq))
        for p in cells:
            added = {ADD[unit][v] for v in options[p]}
            options[p] = options[p] | added if dynamic else added

    goal_at = {
        g["p"]: (1 << i, g["cfg"])
        for i, g in enumerate(m["goals"])
    }
    refill_at = {p: 1 << i for i, p in enumerate(m["refills"])}
    signature = (
        tuple(board),
        tuple((g["p"], g["cfg"]) for g in m["goals"]),
        tuple(m["refills"]),
        tuple(sorted(m["pushers"].items())),
        tuple((p["typ"], tuple(p.get("cycle") or [p["p"]]))
              for p in m["pads"]),
        m["decrement"],
    )
    return {
        "edges": edges, "seqs": sequences, "period": period,
        "options": options, "goal_at": goal_at, "refill_at": refill_at,
        "signature": signature,
    }


def heuristics(m, model, deadline):
    n = len(m["coords"])
    goals = m["goals"]
    k = len(goals)
    reverse = [set() for _ in range(n)]
    opts = model["options"]

    for p, es in enumerate(model["edges"]):
        for _, q, dest, pushing in es:
            deltas = opts[q]
            if pushing:
                deltas = {ADD[a][b] for a in opts[q] for b in opts[dest]}
            for delta in deltas:
                reverse[dest].add((p, delta))
    reverse = [list(v) for v in reverse]

    distances = []
    for goal in goals:
        h = [INF] * (n * 96)
        root = goal["p"] * 96 + goal["cfg"]
        h[root] = 0
        queue = deque([root])
        while queue:
            u = queue.popleft()
            p, cfg = divmod(u, 96)
            nd = h[u] + 1
            for pp, delta in reverse[p]:
                v = pp * 96 + SUB[delta][cfg]
                if h[v] == INF:
                    h[v] = nd
                    queue.append(v)
        distances.append(h)
        if time.monotonic() > deadline:
            return None

    tail_cache = {}

    def tail(g, mask):
        if not mask:
            return 0
        key = (g, mask)
        if key in tail_cache:
            return tail_cache[key]
        index = goals[g]["p"] * 96 + goals[g]["cfg"]
        best = INF
        for j in range(k):
            if mask & (1 << j):
                best = min(best, distances[j][index]
                           + tail(j, mask ^ (1 << j)))
        tail_cache[key] = best
        return best

    full = (1 << k) - 1
    H = {full: [0] * (n * 96)}
    for done in range(full):
        remaining = full ^ done
        h = [INF] * (n * 96)
        for j in range(k):
            if not remaining & (1 << j):
                continue
            extra = tail(j, remaining ^ (1 << j))
            if extra >= INF:
                continue
            for i, d in enumerate(distances[j]):
                v = d + extra
                if v < h[i]:
                    h[i] = v
        H[done] = h
    return H


def search(m, state, model, deadline):
    goals = m["goals"]
    if not goals or len(goals) > 7:
        return [], "unknown"
    full = (1 << len(goals)) - 1
    if state["gm"] == full:
        return [], "unknown"

    H = heuristics(m, model, deadline)
    if H is None:
        return [], "timeout"
    p0, c0 = state["p"], state["cfg"]
    h0 = H[state["gm"]][p0 * 96 + c0]
    if h0 >= INF:
        return [], "unreachable"

    edges = model["edges"]
    n = len(edges)
    period = model["period"]
    goal_at = model["goal_at"]
    refill_at = model["refill_at"]
    maximum = m["maximum"]
    decrement = m["decrement"]
    nG = 1 << len(goals)
    nR = 1 << len(m["refills"])
    effect_cache = {}

    def effects(phase):
        if phase not in effect_cache:
            a = [0] * n
            for typ, seq in model["seqs"]:
                p = seq[phase % len(seq)]
                a[p] = ADD[UNITS[typ]][a[p]]
            effect_cache[phase] = a
        return effect_cache[phase]

    def key(p, c, gm, rm, ph):
        return ((((ph * nR + rm) * nG + gm) * 96 + c) * n + p)

    root = (p0, c0, state["gm"], state["rm"], 0,
            state["energy"], 0, -1, 0)
    nodes = [root]
    dominance = {key(*root[:5]): [(0, root[5])]}
    heap = [(1.15 * h0, -root[5], 0, 0)]
    best_index = 0
    best_rank = (-state["gm"].bit_count(), h0, -root[5], 0)
    solved = None
    expanded = 0
    status = "exhausted"

    while heap:
        _, _, _, ni = heapq.heappop(heap)
        p, c, gm, rm, ph, energy, g, parent, action = nodes[ni]
        if (g, energy) not in dominance.get(key(p, c, gm, rm, ph), ()):
            continue
        if gm == full:
            solved = ni
            status = "solved"
            break
        expanded += 1
        if expanded % 128 == 0 and time.monotonic() > deadline:
            status = "timeout"
            break
        if len(nodes) > 600000:
            status = "timeout"
            break

        nph = (ph + 1) % period
        effect = effects(nph)
        for actnum, q, dest, pushing in edges[p]:
            gate = goal_at.get(q)
            if gate and not gm & gate[0] and c != gate[1]:
                continue

            nc = ADD[effect[q]][c]
            nrm = rm
            bit = refill_at.get(q, 0)
            if bit and not rm & bit:
                ne = maximum
                nrm |= bit
            else:
                ne = energy - decrement
            if ne < 0:
                continue

            ngm = gm
            if pushing:
                nc = ADD[effect[dest]][nc]
                bit = refill_at.get(dest, 0)
                if bit and not nrm & bit:
                    ne = maximum
                    nrm |= bit
            else:
                gate = goal_at.get(dest)
                if gate and nc == gate[1]:
                    ngm |= gate[0]

            ng = g + 1
            h = H[ngm][dest * 96 + nc]
            if h >= INF:
                continue
            nk = key(dest, nc, ngm, nrm, nph)
            records = dominance.get(nk, [])
            if any(og <= ng and oe >= ne for og, oe in records):
                continue
            records = [(og, oe) for og, oe in records
                       if not (ng <= og and ne >= oe)]
            records.append((ng, ne))
            dominance[nk] = records

            nn = len(nodes)
            nodes.append((dest, nc, ngm, nrm, nph, ne, ng, ni, actnum))
            heapq.heappush(heap, (ng + 1.15 * h, -ne, ng, nn))
            rank = (-ngm.bit_count(), h, -ne, ng)
            if rank < best_rank:
                best_rank, best_index = rank, nn

    end = solved if solved is not None else best_index
    path = []
    while nodes[end][7] != -1:
        p, c, gm, rm, ph, energy, g, parent, action = nodes[end]
        poses = [seq[ph % len(seq)] for _, seq in model["seqs"]]
        path.append({
            "action": "ACTION" + str(action),
            "expect": (p, c, gm, rm, energy, poses),
        })
        end = parent
    path.reverse()
    if status != "solved":
        path = path[:3]
    return path, status


def reachable(m, state, model, avoid_special=False):
    root = state["p"]
    queue = deque([root])
    dist = {root: 0}
    first = {}
    cfg, gm = state["cfg"], state["gm"]
    while queue:
        p = queue.popleft()
        for actnum, q, dest, pushing in model["edges"][p]:
            gate = model["goal_at"].get(q)
            if gate and not gm & gate[0] and cfg != gate[1]:
                continue
            if avoid_special and (
                pushing or q in model["goal_at"] or q in model["refill_at"]
            ):
                continue
            if dest not in dist:
                dist[dest] = dist[p] + 1
                first[dest] = actnum if p == root else first[p]
                queue.append(dest)
    return dist, first


def scout(m, state, model):
    dist, first = reachable(m, state, model)
    cfg = state["cfg"]
    required = set()
    for g in m["goals"]:
        if g["done"]:
            continue
        target = g["cfg"]
        if cfg // 16 != target // 16:
            required.add(0)
        if (cfg // 4) % 4 != (target // 4) % 4:
            required.add(1)
        if cfg % 4 != target % 4:
            required.add(2)

    candidates = []
    for pad in m["pads"]:
        q = pad["p"]
        if pad["typ"] in required and q in first:
            candidates.append((dist[q], q))
    for g in m["goals"]:
        q = g["p"]
        if not g["done"] and g["cfg"] == cfg and q in first:
            candidates.append((dist[q], q))
    if candidates:
        _, q = min(candidates)
        return "ACTION" + str(first[q])
    return None


def probe(m, state, model, incomplete):
    dist, first = reachable(m, state, model, True)
    root, coords = state["p"], m["coords"]
    targets = []
    for pad in incomplete:
        hist = [p for _, p in pad.get("hist", [])] or [pad["p"]]
        xx = sum(coords[p][0] for p in hist) / len(hist)
        yy = sum(coords[p][1] for p in hist) / len(hist)
        targets.append((xx, yy))

    if m["fog"] and targets:
        best = None
        for p in first:
            x, y = coords[p]
            separation = max(math.hypot(x - xx, y - yy)
                             for xx, yy in targets)
            score = separation + 0.7 * dist[p]
            if best is None or score < best[0]:
                best = (score, p)
        current_sep = max(
            math.hypot(coords[root][0] - xx, coords[root][1] - yy)
            for xx, yy in targets
        )
        if best and current_sep > 10:
            return "ACTION" + str(first[best[1]])

    candidates = []
    pad_cells = {pad["p"] for pad in m["pads"]}
    for actnum, q, dest, pushing in model["edges"][root]:
        if pushing or q in model["goal_at"] or q in model["refill_at"]:
            continue
        score = 10 if dest == state["oldp"] else 0
        if dest in pad_cells:
            score -= 4
        if m["fog"] and targets:
            x, y = coords[dest]
            score -= max(math.hypot(x - xx, y - yy)
                         for xx, yy in targets) / 5
        candidates.append((score, -actnum, actnum))
    if candidates:
        return "ACTION" + str(max(candidates)[2])
    return None


def explore(m, state, model):
    dist, first = reachable(m, state, model)
    coords = m["coords"]
    energy, decrement = state["energy"], m["decrement"]

    refill_choices = []
    for j, p in enumerate(m["refills"]):
        if not state["rm"] & (1 << j) and p in first:
            if dist[p] * decrement <= energy:
                refill_choices.append((dist[p], p))
    if refill_choices and energy < m["maximum"] * 0.45:
        _, p = min(refill_choices)
        return "ACTION" + str(first[p])

    unknown = [coords[i] for i, v in enumerate(m["board"]) if v < 0]
    best = None
    for p in first:
        if dist[p] * decrement > energy:
            continue
        x, y = coords[p]
        information = sum(
            (x - u) ** 2 + (y - v) ** 2 <= 19 ** 2
            for u, v in unknown
        )
        if information:
            score = information / ((dist[p] + 2) ** 0.75)
            candidate = (score, -dist[p], p)
            if best is None or candidate > best:
                best = candidate
    if best:
        return "ACTION" + str(first[best[2]])

    if refill_choices and energy < m["maximum"] - 4:
        _, p = min(refill_choices)
        return "ACTION" + str(first[p])
    return None


def act(obs, available, memory):
    deadline = time.monotonic() + 8.0
    obs = np.asarray(obs)
    m = memory
    state = read_frame(obs, m)
    model = make_model(m)
    names = set(available)

    def finish(action):
        if action not in names:
            choices = [a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
                       if a in names]
            action = choices[0] if choices else next(iter(names), "ACTION1")
        if action == "RESET":
            m["reset_count"] += 1
            m["plan"] = []
            m.pop("expect", None)
        m["prev"] = {
            "xy": state["xy"], "cfg": state["cfg"],
            "energy": state["energy"], "lives": state["lives"],
            "action": action,
        }
        return action, None, None

    def execute_plan():
        item = m["plan"].pop(0)
        m["expect"] = item["expect"]
        return finish(item["action"])

    expected = m.get("expect")
    valid = False
    if expected is not None and not state["reset"]:
        ep, ec, eg, er, ee, eposes = expected
        valid = (
            state["p"] == ep and state["cfg"] == ec
            and state["gm"] == eg and state["rm"] == er
            and state["energy"] >= ee
            and [pad["p"] for pad in m["pads"]] == list(eposes)
            and m.get("plan_signature") == model["signature"]
        )
    if valid and m.get("plan"):
        return execute_plan()
    m["plan"] = []
    m.pop("expect", None)

    incomplete = [
        p for p in m["pads"]
        if p.get("dynamic") and not p.get("cycle")
    ]
    if incomplete and m.get("probe_steps", 0) < 40:
        action = probe(m, state, model, incomplete)
        if action:
            m["probing"] = True
            m["probe_steps"] = m.get("probe_steps", 0) + 1
            if state["energy"] < m["decrement"] * 2 and "RESET" in names:
                return finish("RESET")
            return finish(action)

    if m.get("probing") and not incomplete:
        m["probing"] = False
        m["probe_steps"] = 0
        if (not m["fog"] and state["energy"] < m["maximum"] - 4
                and m["reset_count"] < 2 and "RESET" in names):
            return finish("RESET")

    if m["calls"] == 1:
        action = scout(m, state, model)
        if action:
            return finish(action)

    plan, status = search(m, state, model, deadline)
    if plan and status == "solved":
        m["plan"] = plan
        m["plan_signature"] = model["signature"]
        return execute_plan()

    if status in ("unreachable", "unknown", "exhausted"):
        action = explore(m, state, model)
        if action:
            return finish(action)

    if (status == "exhausted" and not m["fog"]
            and state["energy"] < m["maximum"] - 3
            and m["reset_count"] < 3 and "RESET" in names):
        return finish("RESET")

    if plan:
        m["plan"] = plan
        m["plan_signature"] = model["signature"]
        return execute_plan()

    action = explore(m, state, model)
    if action:
        return finish(action)
    action = scout(m, state, model)
    if action:
        return finish(action)

    if state["energy"] < m["decrement"] * 2 and "RESET" in names:
        return finish("RESET")

    candidates = []
    visits = m.setdefault("fallback_visits", {})
    for actnum, q, dest, pushing in model["edges"][state["p"]]:
        gate = model["goal_at"].get(q)
        if gate and not state["gm"] & gate[0] and state["cfg"] != gate[1]:
            continue
        candidates.append((visits.get(dest, 0), actnum, dest))
    if candidates:
        _, actnum, dest = min(candidates)
        visits[dest] = visits.get(dest, 0) + 1
        return finish("ACTION" + str(actnum))

    return finish("RESET" if "RESET" in names else "ACTION1")

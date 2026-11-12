import math
import time
import heapq
from array import array
from collections import deque

import numpy as np


# Shape masks, in their unrotated orientations.
_SHAPES = [
    [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
    [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
    [[1, 0, 1], [1, 0, 1], [1, 1, 1]],
    [[0, 1, 1], [1, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
    [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
]
_COLORS = (12, 9, 14, 8)
_GLYPHS = {}
for _s, _shape in enumerate(_SHAPES):
    for _r in range(4):
        _mask = np.rot90(np.array(_shape), -_r)
        _GLYPHS[tuple(int(v) for v in _mask.flat)] = (_s, _r)

# Attribute encoding: shape * 16 + colour * 4 + rotation.
_ADD = []
_SUB = []
for _delta in range(96):
    _ds, _dc, _dr = _delta // 16, (_delta // 4) % 4, _delta % 4
    _forward, _backward = [], []
    for _a in range(96):
        _s, _c, _r = _a // 16, (_a // 4) % 4, _a % 4
        _forward.append(
            ((_s + _ds) % 6) * 16
            + ((_c + _dc) % 4) * 4
            + (_r + _dr) % 4
        )
        _backward.append(
            ((_s - _ds) % 6) * 16
            + ((_c - _dc) % 4) * 4
            + (_r - _dr) % 4
        )
    _ADD.append(_forward)
    _SUB.append(_backward)

_DIRS = ((0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4))
_INF = 10**8


def _decode(patch):
    """Decode a coloured 3x3 shape."""
    best = None
    for ci, color in enumerate(_COLORS):
        mask = tuple(int(v == color) for v in patch.flat)
        sr = _GLYPHS.get(mask)
        if sr is not None:
            s, r = sr
            best = s * 16 + ci * 4 + r
            break
    return best


def _find_player(obs):
    # The avatar is a solid 5x5 rectangle: two pink rows, three orange rows.
    candidates = np.argwhere(
        (obs[:60, :60] == 12)
        & (obs[1:61, :60] == 12)
        & (obs[2:62, :60] == 9)
        & (obs[4:64, :60] == 9)
    )
    for y, x in candidates:
        y, x = int(y), int(x)
        p = obs[y:y + 5, x:x + 5]
        if p.shape == (5, 5):
            if np.all(p[:2] == 12) and np.all(p[2:] == 9):
                return x, y
    return None


def _xy(st, cell):
    return st["ox"] + (cell % st["nx"]) * 5, (cell // st["nx"]) * 5


def _cell(st, x, y):
    c = (x - st["ox"]) // 5
    r = y // 5
    if 0 <= c < st["nx"] and 0 <= r < st["ny"]:
        return r * st["nx"] + c
    return None


def _neighbor(st, cell, dx, dy):
    c, r = cell % st["nx"], cell // st["nx"]
    c += dx
    r += dy
    if 0 <= c < st["nx"] and 0 <= r < st["ny"]:
        return r * st["nx"] + c
    return None


def _distance(st, a, b):
    return (
        abs(a % st["nx"] - b % st["nx"])
        + abs(a // st["nx"] - b // st["nx"])
    )


def _infer_cycle(track):
    hist = track["hist"]
    if not hist:
        return
    old = track.get("cycle")
    if old and all(old[t % len(old)] == p for t, p in hist.items()):
        return

    track["cycle"] = None
    if len(set(hist.values())) <= 1:
        return

    t = max(hist)
    for period in range(2, 41):
        if not all(k in hist for k in (t, t - 1, t - period, t - period - 1)):
            continue
        if hist[t] != hist[t - period] or hist[t - 1] != hist[t - period - 1]:
            continue
        if not all(k in hist for k in range(t - period + 1, t + 1)):
            continue
        cycle = [None] * period
        for k in range(t - period + 1, t + 1):
            cycle[k % period] = hist[k]
        if all(cycle[k % period] == p for k, p in hist.items()):
            track["cycle"] = cycle
            return


def _update_tracks(st, detections):
    tick = st["tick"]
    tracks = st["tracks"]

    for kind in range(3):
        seen = list(detections[kind])
        ids = [i for i, tr in enumerate(tracks) if tr["kind"] == kind]
        pairs = []
        for i in ids:
            tr = tracks[i]
            cyc = tr.get("cycle")
            predicted = cyc[tick % len(cyc)] if cyc else tr["pos"]
            for j, pos in enumerate(seen):
                pairs.append((_distance(st, predicted, pos), i, j))
        pairs.sort()

        used_i, used_j = set(), set()
        for dist, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            tr = tracks[i]
            elapsed = max(1, tick - tr.get("seen", tick))
            if dist > max(2, elapsed + 1):
                continue
            used_i.add(i)
            used_j.add(j)
            tr["pos"] = seen[j]
            tr["seen"] = tick
            tr["hist"][tick] = seen[j]
            if len(tr["hist"]) > 90:
                oldest = sorted(tr["hist"])[:-80]
                for k in oldest:
                    del tr["hist"][k]
            _infer_cycle(tr)

        for j, pos in enumerate(seen):
            if j not in used_j:
                tracks.append({
                    "kind": kind,
                    "pos": pos,
                    "seen": tick,
                    "hist": {tick: pos},
                    "cycle": None,
                })


def _observe(obs, st, player):
    px, py = player
    pos = _cell(st, px, py)
    ui = obs[np.ix_([55, 57, 59], [3, 5, 7])]
    attr = _decode(ui)
    if attr is None:
        attr = st.get("attr", 0)

    bar = []
    for x in range(13, 56):
        if int(obs[61, x]) not in (3, 11):
            break
        bar.append(int(obs[61, x]))
    capacity = len(bar) if bar else 42
    energy = bar.count(11) if bar else st.get("energy", capacity)
    lives = sum(int(obs[61, x]) == 8 for x in (56, 59, 62))

    restarted = st.pop("reset_pending", False)
    if "lives" in st and lives < st["lives"]:
        restarted = True

    if restarted:
        st["tick"] = 0
        for k, v in list(st["goals"].items()):
            st["goals"][k] = (v[0], True)
        for k in st["refills"]:
            st["refills"][k] = True
        for tr in st["tracks"]:
            tr["hist"] = {}
            tr["cycle"] = None
            tr["seen"] = 0
        st.pop("plan", None)
        st.pop("model", None)
        st["round_steps"] = 0
    elif "pos" in st and st.get("last_action", "").startswith("ACTION"):
        moved = pos != st["pos"]
        if not moved and st.get("expected"):
            ex = st["expected"]
            moved = ex[0] == pos and st.get("expected_stationary_move", False)
        if moved:
            st["tick"] += 1

        diff = st.get("energy", energy) - energy
        if diff in (1, 2, 3) and not restarted:
            st["decrement"] = diff

    fog = int(np.count_nonzero(obs[:50, 4:] == 5)) > 500
    st["fog"] = fog

    def visible(x, y, margin=0):
        if not (0 <= x - margin and x + margin < 64
                and 0 <= y - margin and y + margin < 60):
            return False
        if not fog:
            return True
        cx, cy = px + 1.5, py + 1.5
        return all(
            (xx - cx) ** 2 + (yy - cy) ** 2 <= 399
            for xx in (x - margin, x + margin)
            for yy in (y - margin, y + margin)
        )

    detections = [set(), set(), set()]
    found_goals = {}
    found_refills = set()
    fully_seen = set()

    for cell in range(st["nx"] * st["ny"]):
        x, y = _xy(st, cell)
        if x < 12 and y >= 50:
            st["grid"][cell] = 2
            continue

        if visible(x + 2, y + 2):
            center = int(obs[y + 2, x + 2])
            if center == 4:
                st["grid"][cell] = 2
            elif center in (0, 1, 3, 8, 9, 11, 12, 14):
                st["grid"][cell] = 1

        if not visible(x + 2, y + 2, 1):
            continue
        fully_seen.add(cell)
        if cell == pos:
            continue

        patch = obs[y + 1:y + 4, x + 1:x + 4]
        if patch.shape != (3, 3):
            continue

        ga = _decode(patch)
        if ga is not None:
            color = _COLORS[(ga // 4) % 4]
            if np.all((patch == color) | (patch == 5)):
                found_goals[cell] = ga
                st["all_goals"].add(cell)
                st["grid"][cell] = 1
                continue

        ring = patch.copy()
        ring[1, 1] = 11
        if np.all(ring == 11):
            found_refills.add(cell)
            st["grid"][cell] = 1
            continue

        colors_present = set(int(v) for v in patch.flat)
        zeros = int(np.count_nonzero(patch == 0))
        ones = int(np.count_nonzero(patch == 1))
        if 0 in colors_present and {8, 9, 12, 14}.issubset(colors_present):
            detections[1].add(cell)
            st["grid"][cell] = 1
        elif zeros >= 3 and ones >= 2:
            detections[2].add(cell)
            st["grid"][cell] = 1
        elif zeros >= 3 and ones == 0:
            detections[0].add(cell)
            st["grid"][cell] = 1

        # A completed goal may retain its black frame.
        if visible(x + 2, y + 2, 3) and x >= 1 and y >= 1:
            border = obs[y - 1:y + 6, x - 1:x + 6]
            if border.shape == (7, 7):
                if (np.all(border[0] == 5) and np.all(border[-1] == 5)
                        and np.all(border[:, 0] == 5)
                        and np.all(border[:, -1] == 5)):
                    if np.any(border[1:6, 1:6] != 5):
                        st["all_goals"].add(cell)
                        st["grid"][cell] = 1

    for cell, ga in found_goals.items():
        st["goals"][cell] = (ga, True)
    for cell, (ga, active) in list(st["goals"].items()):
        if cell == pos and attr == ga:
            st["goals"][cell] = (ga, False)
        elif active and cell in fully_seen and cell != pos and cell not in found_goals:
            st["goals"][cell] = (ga, False)

    for cell in found_refills:
        st["refills"][cell] = True
    for cell in list(st["refills"]):
        if cell == pos:
            st["refills"][cell] = False
        elif cell in fully_seen and cell not in found_refills:
            st["refills"][cell] = False

    # Thin blue stripes are launch pads; they point away from the wall.
    for cell in range(st["nx"] * st["ny"]):
        if st["grid"][cell] == 2:
            continue
        x, y = _xy(st, cell)
        tests = [
            (x - 1, y, x - 1, y + 4, 1, 0),
            (x + 5, y, x + 5, y + 4, -1, 0),
            (x, y - 1, x + 4, y - 1, 0, 1),
            (x, y + 5, x + 4, y + 5, 0, -1),
        ]
        for x1, y1, x2, y2, dx, dy in tests:
            if not (visible(x1, y1) and visible(x2, y2)):
                continue
            stripe = obs[y1:y2 + 1, x1:x2 + 1]
            if stripe.size == 5 and np.all(stripe == 1):
                st["bumpers"][cell] = (dx, dy)
                st["grid"][cell] = 1

    st["grid"][pos] = 1
    _update_tracks(st, detections)

    st["pos"], st["attr"] = pos, attr
    st["energy"], st["capacity"], st["lives"] = energy, capacity, lives
    st["round_steps"] = st.get("round_steps", 0) + 1
    st["visits"][pos] = st["visits"].get(pos, 0) + 1


def _switch_models(st):
    result = []
    period = 1
    for tr in st["tracks"]:
        cyc = tr.get("cycle")
        if cyc:
            proposed = math.lcm(period, len(cyc))
            if proposed <= 64:
                period = proposed
                result.append((tr["kind"], tuple(cyc)))
            else:
                result.append((tr["kind"], (cyc[st["tick"] % len(cyc)],)))
        else:
            result.append((tr["kind"], (tr["pos"],)))
    return tuple(result), period


def _landing(st, cell):
    direction = st["bumpers"].get(cell)
    if direction is None:
        return cell
    dx, dy = direction
    for distance in range(1, 12):
        q = _neighbor(st, cell, dx * distance, dy * distance)
        if q is not None:
            if st["grid"][q] == 2 or q in st["all_goals"]:
                return _neighbor(st, cell, dx * (distance - 1), dy * (distance - 1))
    return cell


def _model(st, deadline):
    switches, period = _switch_models(st)
    goals = tuple(sorted(
        (cell, ga) for cell, (ga, active) in st["goals"].items() if active
    ))
    refills = tuple(sorted(cell for cell, active in st["refills"].items() if active))
    sig = (
        tuple(st["grid"]),
        tuple(sorted(st["bumpers"].items())),
        tuple(sorted(st["all_goals"])),
        goals, refills, switches, period, st["decrement"],
    )
    old = st.get("model")
    if old is not None and old["sig"] == sig:
        return old

    free = {i for i, v in enumerate(st["grid"]) if v == 1}
    for cell in st["bumpers"]:
        q = _landing(st, cell)
        if q is not None and st["grid"][q] != 2:
            free.add(q)
    free.add(st["pos"])
    cells = sorted(free)
    index = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    size = n * period * 96
    goal_index = {c: i for i, (c, a) in enumerate(goals)}
    refill_index = {c: 1 << i for i, c in enumerate(refills)}

    at = []
    for phase in range(period):
        contents = {}
        for kind, cyc in switches:
            c = cyc[(phase + 1) % len(cyc)]
            vec = contents.setdefault(c, [0, 0, 0])
            vec[kind] += 1
        at.append(contents)

    adj = [[] for _ in range(n * period)]
    reverse = [[] for _ in range(n * period)]

    for phase in range(period):
        next_phase = (phase + 1) % period
        for i, source in enumerate(cells):
            src_tp = phase * n + i
            for dx, dy, action in _DIRS:
                entry = _neighbor(st, source, dx, dy)
                if entry not in index:
                    continue
                end = _landing(st, entry)
                if end not in index:
                    continue
                delta = list(at[phase].get(entry, (0, 0, 0)))
                pushed = end != entry
                if pushed:
                    extra = at[phase].get(end, (0, 0, 0))
                    delta = [delta[k] + extra[k] for k in range(3)]
                di = (delta[0] % 6) * 16 + (delta[1] % 4) * 4 + delta[2] % 4
                dst_tp = next_phase * n + index[end]
                r1 = refill_index.get(entry, 0)
                r2 = refill_index.get(end, 0) if pushed else 0
                gi = goal_index.get(entry, -1)
                edge = (dst_tp * 96, di, r1, r2, gi, action, end, pushed)
                adj[src_tp].append(edge)
                reverse[dst_tp].append((src_tp * 96, di))

    model = {
        "sig": sig, "period": period, "cells": cells, "index": index,
        "n": n, "size": size, "adj": adj, "goals": goals,
        "refills": refills, "dist": None, "tails": None,
    }

    # Reverse pattern databases: exact distances to each goal when energy
    # and the locks on other goals are relaxed.
    distances = []
    complete = True
    for goal_cell, goal_attr in goals:
        dist = array("i", [-1]) * size
        queue = deque()
        if goal_cell not in index:
            complete = False
            break
        for phase in range(period):
            b = (phase * n + index[goal_cell]) * 96 + goal_attr
            dist[b] = 0
            queue.append(b)
        count = 0
        while queue:
            b = queue.popleft()
            tp, a = divmod(b, 96)
            nd = dist[b] + 1
            for src96, di in reverse[tp]:
                nb = src96 + _SUB[di][a]
                if dist[nb] < 0:
                    dist[nb] = nd
                    queue.append(nb)
            count += 1
            if count % 8192 == 0 and time.perf_counter() > deadline:
                complete = False
                break
        if not complete:
            break
        distances.append(dist)

    if complete and len(distances) == len(goals):
        model["dist"] = distances
        ng = len(goals)
        pair = [[_INF] * ng for _ in range(ng)]
        for i, (cell, a) in enumerate(goals):
            for j in range(ng):
                vals = [
                    distances[j][(p * n + index[cell]) * 96 + a]
                    for p in range(period)
                ]
                vals = [v for v in vals if v >= 0]
                if vals:
                    pair[i][j] = min(vals)

        # Held-Karp lower bounds for visiting the remaining goals.
        if ng <= 7:
            tails = [[_INF] * ng for _ in range(1 << ng)]
            for i in range(ng):
                tails[1 << i][i] = 0
            for mask in range(1, 1 << ng):
                for i in range(ng):
                    if not (mask & (1 << i)):
                        continue
                    rest = mask ^ (1 << i)
                    if not rest:
                        continue
                    tails[mask][i] = min(
                        pair[i][j] + tails[rest][j]
                        for j in range(ng) if rest & (1 << j)
                    )
            model["tails"] = tails

    st["model"] = model
    return model


def _heuristic(model, base, completed):
    ng = len(model["goals"])
    remaining = ((1 << ng) - 1) ^ completed
    if not remaining:
        return 0
    distances = model["dist"]
    if distances is None:
        return 0

    tails = model["tails"]
    if tails is not None:
        best = _INF
        for i in range(ng):
            if remaining & (1 << i):
                d = distances[i][base]
                if d >= 0:
                    best = min(best, d + tails[remaining][i])
        return best

    values = []
    for i in range(ng):
        if remaining & (1 << i):
            d = distances[i][base]
            if d < 0:
                return _INF
            values.append(d)
    return max(values, default=0)


def _path(nodes, end, model):
    result = []
    n = model["n"]
    while nodes[end][5] >= 0:
        base, completed, used, fuel, g, parent, action = nodes[end]
        tp, attr = divmod(base, 96)
        phase, i = divmod(tp, n)
        result.append((action, model["cells"][i], attr, phase, fuel))
        end = parent
    result.reverse()
    return result


def _search(st, model, deadline):
    goals = model["goals"]
    if not goals:
        return None, False
    n, size = model["n"], model["size"]
    period = model["period"]
    phase = st["tick"] % period
    root = (phase * n + model["index"][st["pos"]]) * 96 + st["attr"]
    d = max(1, st["decrement"])
    fuel = st["energy"] // d
    capacity = st["capacity"] // d
    ng = len(goals)
    full = (1 << ng) - 1
    masks = 1 << ng
    h0 = _heuristic(model, root, 0)
    if h0 >= _INF:
        return None, False

    # Labels are Pareto-optimal (elapsed actions, energy remaining).
    nodes = [(root, 0, 0, fuel, 0, -1, 0)]
    labels = {root: [(0, fuel)]}
    heap = [(h0, h0, -fuel, 0)]
    best_id, best_h = 0, h0
    exhausted = True
    expansions = 0

    while heap:
        _, h, _, node_id = heapq.heappop(heap)
        base, completed, used, fuel, g, parent, action = nodes[node_id]
        key = base + size * (completed + masks * used)
        if (g, fuel) not in labels.get(key, ()):
            continue
        if completed == full:
            return _path(nodes, node_id, model), True
        if h < best_h:
            best_h, best_id = h, node_id

        tp, attr = divmod(base, 96)
        for dst96, di, r1, r2, gi, ac, end, pushed in model["adj"][tp]:
            if gi >= 0 and not (completed & (1 << gi)):
                if attr != goals[gi][1]:
                    continue

            na = _ADD[di][attr]
            nb = dst96 + na
            nc = completed
            if gi >= 0 and na == goals[gi][1]:
                nc |= 1 << gi

            nf = capacity if r1 & ~used else fuel - 1
            if pushed and nf < 0:
                continue
            if r2 & ~used:
                nf = capacity
            if nf < 0 and nc != full:
                continue
            nu = used | r1 | r2
            ngcost = g + 1
            nh = _heuristic(model, nb, nc)
            if nh >= _INF:
                continue

            nk = nb + size * (nc + masks * nu)
            old = labels.get(nk)
            if old is not None:
                if any(og <= ngcost and of >= nf for og, of in old):
                    continue
                old = [
                    (og, of) for og, of in old
                    if not (ngcost <= og and nf >= of)
                ]
            else:
                old = []
            old.append((ngcost, nf))
            labels[nk] = old

            ni = len(nodes)
            nodes.append((nb, nc, nu, nf, ngcost, node_id, ac))
            heapq.heappush(heap, (ngcost + nh, nh, -nf, ni))

        expansions += 1
        if expansions % 1024 == 0:
            if time.perf_counter() > deadline or len(nodes) > 240000:
                exhausted = False
                break

    if not exhausted and best_id and best_h < h0:
        return _path(nodes, best_id, model), False
    return None, False


def _explore(st, model, deadline):
    """Find a safe route to a useful viewpoint, refill, or little-visited cell."""
    n = model["n"]
    phase = st["tick"] % model["period"]
    root = (phase * n + model["index"][st["pos"]]) * 96 + st["attr"]
    decrement = max(1, st["decrement"])
    initial_fuel = st["energy"] // decrement
    capacity = st["capacity"] // decrement

    unknown = [i for i, v in enumerate(st["grid"]) if v == 0]
    gains = {}
    for cell in model["cells"]:
        x, y = _xy(st, cell)
        gain = 0
        for u in unknown:
            ux, uy = _xy(st, u)
            if (ux - x) ** 2 + (uy - y) ** 2 <= 17 ** 2:
                gain += 1
        gains[cell] = gain

    nodes = [(root, 0, 0, initial_fuel, 0, -1, 0)]
    queue = deque([0])
    visited = {root}
    best_id, best_score, best_gain = None, -1e30, 0

    while queue:
        ni = queue.popleft()
        base, completed, used, fuel, depth, parent, action = nodes[ni]
        tp, attr = divmod(base, 96)
        cell = model["cells"][tp % n]
        if depth:
            gain = gains.get(cell, 0)
            if gain:
                score = 100.0 * gain / (depth + 3)
            else:
                score = 1.0 / (1 + st["visits"].get(cell, 0)) - 0.065 * depth
                if fuel > initial_fuel:
                    score += 4 + (fuel - initial_fuel) * 0.1
            if score > best_score:
                best_score, best_id, best_gain = score, ni, gain

        if depth >= 22:
            continue
        for dst96, di, r1, r2, gi, ac, end, pushed in model["adj"][tp]:
            if gi >= 0 and not (completed & (1 << gi)):
                if attr != model["goals"][gi][1]:
                    continue
            na = _ADD[di][attr]
            nb = dst96 + na
            if nb in visited:
                continue
            nf = capacity if r1 & ~used else fuel - 1
            if pushed and nf < 0:
                continue
            if r2 & ~used:
                nf = capacity
            if nf < 0:
                continue
            nc = completed
            if gi >= 0 and na == model["goals"][gi][1]:
                nc |= 1 << gi
            visited.add(nb)
            idx = len(nodes)
            nodes.append((nb, nc, used | r1 | r2, nf, depth + 1, ni, ac))
            queue.append(idx)

        if len(nodes) > 10000:
            break
        if len(nodes) % 256 == 0 and time.perf_counter() > deadline:
            break

    if best_id is None:
        return None, 0
    return _path(nodes, best_id, model), best_gain


def act(obs, available, memory):
    started = time.perf_counter()
    deadline = started + 8.7
    obs = np.asarray(obs)
    player = _find_player(obs)

    def fallback():
        for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if a in available:
                return a, None, None
        a = available[0] if available else "ACTION1"
        return a, None, None

    if player is None:
        return fallback()

    if "ls20" not in memory:
        ox = player[0] % 5
        nx = (64 - ox) // 5
        memory["ls20"] = {
            "ox": ox, "nx": nx, "ny": 12,
            "grid": [0] * (nx * 12),
            "goals": {}, "all_goals": set(), "refills": {},
            "bumpers": {}, "tracks": [], "tick": 0,
            "decrement": 1, "visits": {}, "resets": 0,
        }
    st = memory["ls20"]
    _observe(obs, st, player)

    # Reuse a previous plan only when both perception and its predicted
    # physical state still agree.
    model = _model(st, min(deadline - 2.0, started + 4.5))
    expected = st.get("expected")
    plan = st.get("plan")
    valid_plan = (
        plan
        and st.get("plan_sig") == model["sig"]
        and expected is not None
        and expected[0] == st["pos"]
        and expected[1] == st["attr"]
        and expected[2] == st["tick"] % model["period"]
        and st["energy"] // max(1, st["decrement"]) >= expected[3]
    )

    if not valid_plan:
        plan, solved = _search(st, model, deadline - 0.6)
        st["plan"] = plan or []
        st["plan_sig"] = model["sig"]

        if not plan:
            exploration, gain = _explore(st, model, deadline - 0.1)
            low_energy = st["energy"] < st["capacity"] - 8
            can_reset = (
                "RESET" in available
                and st.get("resets", 0) < 2
                and st.get("round_steps", 0) > 10
            )
            if can_reset and low_energy and gain == 0:
                st["resets"] += 1
                st["reset_pending"] = True
                st["last_action"] = "RESET"
                st["plan"] = []
                st.pop("expected", None)
                return "RESET", None, None
            plan = exploration or []
            # Exploration is deliberately receding-horizon.
            st["plan"] = plan[:1]
            st["plan_sig"] = model["sig"]

    plan = st.get("plan", [])
    if plan:
        action, next_pos, next_attr, next_phase, next_fuel = plan.pop(0)
        name = "ACTION" + str(action)
        if name in available:
            st["expected"] = (next_pos, next_attr, next_phase, next_fuel)
            st["expected_stationary_move"] = next_pos == st["pos"]
            st["last_action"] = name
            return name, None, None

    st["plan"] = []
    st.pop("expected", None)
    if st["energy"] <= 1 and "RESET" in available and st.get("resets", 0) < 2:
        st["resets"] += 1
        st["reset_pending"] = True
        st["last_action"] = "RESET"
        return "RESET", None, None

    # If the current life is exhausted, one ordinary action lets the
    # simulator restore the level while preserving our discovered map.
    choice = fallback()
    st["last_action"] = choice[0]
    return choice

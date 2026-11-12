import numpy as np
import time
from collections import deque


def _components(cells):
    cells = set(cells)
    out = []
    while cells:
        start = cells.pop()
        comp = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for p in ((x - 1, y), (x + 1, y),
                      (x, y - 1), (x, y + 1)):
                if p in cells:
                    cells.remove(p)
                    comp.add(p)
                    stack.append(p)
        out.append(comp)
    return out


def _read_grid(obs, m):
    if "n" not in m:
        # Prefer the game's usual camera size; infer another square camera
        # if its pixel-block alignment fits substantially better.
        candidates = [20] + [n for n in range(10, 33) if n != 20]
        best = None
        for n in candidates:
            s = 64 // n
            off = (64 - n * s) // 2
            a = obs[off:off + n * s, off:off + n * s]
            blocks = a.reshape(n, s, n, s).transpose(0, 2, 1, 3)
            centers = blocks[:, :, s // 2, s // 2]
            mismatch = blocks != centers[:, :, None, None]
            # The step meter can overwrite part of the first block row.
            mismatch[0] = False
            score = int(mismatch.sum())
            if best is None or score < best[0]:
                best = (score, n, s, off)
            if n == 20 and score == 0:
                break
        _, m["n"], m["scale"], m["off"] = best
    n, s, off = m["n"], m["scale"], m["off"]
    idx = off + np.arange(n) * s + s // 2
    return obs[np.ix_(idx, idx)].astype(int)


def _shape_key(sh):
    return sh["w"], sh["h"], sh["mask"], sh["ports"]


def _capture(g, m):
    bg = m["bg"]
    if m["grey"]:
        yy, xx = np.where((g != bg) & (g != 4))
        cells = set(zip(xx.tolist(), yy.tolist()))
        if not cells:
            return None
    else:
        yy, xx = np.where((g == 0) | (g == 8) | (g == 3))
        comps = _components(zip(xx.tolist(), yy.tolist()))
        comps = [c for c in comps if any(g[y, x] == 0 for x, y in c)]
        if not comps:
            return None
        cells = max(comps, key=lambda c: sum(g[y, x] == 0 for x, y in c))

    x0 = min(x for x, y in cells)
    y0 = min(y for x, y in cells)
    x1 = max(x for x, y in cells)
    y1 = max(y for x, y in cells)
    mask = tuple(sorted((x - x0, y - y0) for x, y in cells))
    ports = tuple(sorted((x - x0, y - y0) for x, y in cells
                         if g[y, x] in (8, 3)))
    special = (
        m["grey"]
        and any(g[y, x] == 0 for x, y in cells)
        and any(g[y, x] not in (0, 3, 8) for x, y in cells)
    )
    return {
        "x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
        "mask": mask, "ports": ports, "special": special,
    }


def _geom(o, v=None, r=None):
    if v is None:
        v = o["v"]
    if r is None:
        r = o["r"]
    sh = o["variants"][v]
    w, h = sh["w"], sh["h"]
    mask, ports = sh["mask"], sh["ports"]
    for _ in range(r % 4):
        mask = tuple((h - 1 - y, x) for x, y in mask)
        ports = tuple((h - 1 - y, x) for x, y in ports)
        w, h = h, w
    return w, h, mask, ports


def _same_capture(sh, geom):
    w, h, mask, ports = geom
    return (sh is not None and sh["w"] == w and sh["h"] == h
            and set(sh["mask"]) == set(mask)
            and set(sh["ports"]) == set(ports))


def _absolute(o):
    return {(o["x"] + x, o["y"] + y) for x, y in _geom(o)[2]}


def _identify(g, m, sh=None):
    objects = m["objects"]
    if sh is None:
        sh = _capture(g, m)
    if sh is not None:
        for i, o in enumerate(objects):
            if (o["x"] == sh["x"] and o["y"] == sh["y"]
                    and _same_capture(sh, _geom(o))):
                return i

    if not m["grey"]:
        # A touching connector can enlarge the extracted black component.
        best, score = None, 0.0
        for i, o in enumerate(objects):
            _, _, mask, ports = _geom(o)
            portset = set(ports)
            body = [(o["x"] + x, o["y"] + y)
                    for x, y in mask if (x, y) not in portset]
            if not body:
                continue
            hits = sum(0 <= x < m["n"] and 0 <= y < m["n"]
                       and g[y, x] == 0 for x, y in body)
            fraction = hits / len(body)
            value = fraction + min(hits, 30) * 0.001
            if fraction > 0.65 and value > score:
                best, score = i, value
        return best
    return None


def _new_object(sh):
    return {
        "x": sh["x"], "y": sh["y"],
        "variants": [sh], "v": 0, "r": 0, "family": False,
    }


def _begin_probe(m, i):
    o = m["objects"][i]
    m["phase"] = "probe"
    m["probe"] = {"i": i, "path": [o["v"]], "edges": set(), "count": 0}


def _finish_probe(m):
    p = m["probe"]
    o = m["objects"][p["i"]]
    old_variants = o["variants"]
    order = sorted(range(len(old_variants)),
                   key=lambda k: (len(old_variants[k]["mask"]),
                                  old_variants[k]["h"],
                                  old_variants[k]["w"], k))
    remap = {old: new for new, old in enumerate(order)}
    o["variants"] = [old_variants[k] for k in order]
    o["v"] = remap[o["v"]]
    o["r"] = 0
    o["family"] = True
    if len(p["path"]) >= 2:
        a, b = remap[p["path"][-2]], remap[p["path"][-1]]
        if a != b:
            m["direction"] = 1 if b > a else -1
    m["phase"] = "discover"
    m.pop("probe", None)
    m.pop("solutions", None)
    m.pop("target", None)
    m["failed"] = set()


def _family_next(v, direction, count):
    if count <= 1:
        return v, direction
    nxt = v + direction
    if nxt < 0 or nxt >= count:
        direction = -direction
        nxt = v + direction
    return nxt, direction


def _family_steps(o, target, direction):
    v = o["v"]
    if v == target:
        return 0
    for steps in range(1, 2 * len(o["variants"]) + 3):
        v, direction = _family_next(v, direction, len(o["variants"]))
        if v == target:
            return steps
    return 100


def _turn_cost(o, v, r, m):
    if o["family"]:
        return _family_steps(o, v, m["direction"])
    return (r - o["r"]) % 4


def _shift_and_cost(config, m):
    objects, n = m["objects"], m["n"]
    lo_x = min(c[2] for c in config)
    lo_y = min(c[3] for c in config)
    hi_x = max(c[2] + _geom(objects[i], c[0], c[1])[0]
               for i, c in enumerate(config))
    hi_y = max(c[3] + _geom(objects[i], c[0], c[1])[1]
               for i, c in enumerate(config))
    if hi_x - lo_x > n or hi_y - lo_y > n:
        return None

    dx = sorted(objects[i]["x"] - c[2] for i, c in enumerate(config))
    dy = sorted(objects[i]["y"] - c[3] for i, c in enumerate(config))
    sx = max(-lo_x, min(n - hi_x, dx[len(dx) // 2]))
    sy = max(-lo_y, min(n - hi_y, dy[len(dy) // 2]))
    target, cost = [], 0
    for i, (v, r, x, y) in enumerate(config):
        o = objects[i]
        tx, ty = x + sx, y + sy
        turns = _turn_cost(o, v, r, m)
        distance = abs(tx - o["x"]) + abs(ty - o["y"])
        changed = turns or distance
        cost += turns + distance
        if changed and i != m.get("selected"):
            cost += 1
        target.append((v, r, tx, ty))
    return cost, target


def _solve(m):
    objects, n = m["objects"], m["n"]
    count = len(objects)
    if not count:
        return []
    deadline = time.perf_counter() + 6.3
    failed = m["failed"]
    geometries = []
    for i, o in enumerate(objects):
        gs = []
        for v in range(len(o["variants"])):
            rotations = [0] if o["family"] else range(4)
            if not o["variants"][v]["ports"]:
                rotations = [o["r"]]
            for r in rotations:
                w, h, mask, ports = _geom(o, v, r)
                gs.append((v, r, w, h, ports, _turn_cost(o, v, r, m)))
        geometries.append(gs)

    max_ports = [max(len(g[4]) for g in gs) for gs in geometries]
    anchor = max(range(count), key=lambda i: max_ports[i])
    placements = [None] * count
    results = {}
    cutoff = [float("inf")]
    nodes = [0]
    stopped = [False]

    def lower_bound(bounds, turn_sum):
        minx, miny, maxx, maxy = bounds
        xs, ys = [], []
        for i, p in enumerate(placements):
            if p is not None:
                xs.append(objects[i]["x"] - p[2])
                ys.append(objects[i]["y"] - p[3])
        xs.sort()
        ys.sort()
        sx = max(-minx, min(n - maxx, xs[len(xs) // 2]))
        sy = max(-miny, min(n - maxy, ys[len(ys) // 2]))
        return (turn_sum + sum(abs(x - sx) for x in xs)
                + sum(abs(y - sy) for y in ys))

    def record(edges):
        sig = tuple(sorted(tuple(sorted(edge)) for edge in edges))
        if sig in failed:
            return
        config = tuple(placements)
        scored = _shift_and_cost(config, m)
        if scored is None:
            return
        cost = scored[0]
        old = results.get(sig)
        if old is None or cost < old[0]:
            results[sig] = (cost, config)
        if results:
            best = min(v[0] for v in results.values())
            cutoff[0] = min(cutoff[0], best + 28)
        if len(results) > 500:
            keep = sorted(results, key=lambda k: results[k][0])[:250]
            new = {k: results[k] for k in keep}
            results.clear()
            results.update(new)
            cutoff[0] = min(cutoff[0], max(v[0] for v in results.values()) + 2)

    def dfs(remaining, portmap, open_count, bounds, edges, turn_sum):
        nodes[0] += 1
        if nodes[0] % 128 == 0:
            if time.perf_counter() > deadline or nodes[0] > 160000:
                stopped[0] = True
        if stopped[0]:
            return
        if open_count > sum(max_ports[i] for i in remaining):
            return
        if lower_bound(bounds, turn_sum) > cutoff[0]:
            return
        if not remaining:
            if open_count == 0:
                record(edges)
            return

        parity_options = {0}
        for i in remaining:
            parities = {len(g[4]) % 2 for g in geometries[i]}
            parity_options = {a ^ b for a in parity_options for b in parities}
        if open_count % 2 not in parity_options:
            return

        if open_count:
            point = next(p for p, endpoint in portmap.items()
                         if endpoint is not None)
            px, py = point
            candidates = []
            for i in remaining:
                for v, r, w, h, ports, turns in geometries[i]:
                    for ox, oy in ports:
                        x, y = px - ox, py - oy
                        minx = min(bounds[0], x)
                        miny = min(bounds[1], y)
                        maxx = max(bounds[2], x + w)
                        maxy = max(bounds[3], y + h)
                        if maxx - minx > n or maxy - miny > n:
                            continue
                        matches = 0
                        valid = True
                        for ax, ay in ports:
                            p = (x + ax, y + ay)
                            if p in portmap:
                                if portmap[p] is None:
                                    valid = False
                                    break
                                matches += 1
                        if not valid:
                            continue
                        new_open = open_count + len(ports) - 2 * matches
                        future = sum(max_ports[j] for j in remaining if j != i)
                        if new_open > future:
                            continue
                        ax = objects[anchor]["x"]
                        ay = objects[anchor]["y"]
                        estimate = (abs(objects[i]["x"] - ax - x)
                                    + abs(objects[i]["y"] - ay - y)
                                    + turns - 18 * matches)
                        candidates.append(
                            (estimate, i, v, r, x, y, w, h, ports, turns,
                             (minx, miny, maxx, maxy), new_open)
                        )
            candidates.sort(key=lambda c: c[0])
        else:
            # Another independently connected group can remain.
            i = max(remaining, key=lambda j: max_ports[j])
            candidates = []
            for v, r, w, h, ports, turns in geometries[i]:
                x = objects[i]["x"] - objects[anchor]["x"]
                y = objects[i]["y"] - objects[anchor]["y"]
                b = (min(bounds[0], x), min(bounds[1], y),
                     max(bounds[2], x + w), max(bounds[3], y + h))
                if b[2] - b[0] > n or b[3] - b[1] > n:
                    continue
                if any((x + a, y + b_) in portmap for a, b_ in ports):
                    continue
                candidates.append(
                    (turns, i, v, r, x, y, w, h, ports, turns, b, len(ports))
                )

        for c in candidates:
            if stopped[0]:
                return
            _, i, v, r, x, y, w, h, ports, turns, b, new_open = c
            newmap = portmap.copy()
            newedges = list(edges)
            valid = True
            for k, (ox, oy) in enumerate(ports):
                p = (x + ox, y + oy)
                endpoint = (i, v, k)
                if p in newmap:
                    if newmap[p] is None:
                        valid = False
                        break
                    newedges.append((newmap[p], endpoint))
                    newmap[p] = None
                else:
                    newmap[p] = endpoint
            if not valid:
                continue
            placements[i] = (v, r, x, y)
            dfs(tuple(j for j in remaining if j != i), newmap, new_open,
                b, newedges, turn_sum + turns)
            placements[i] = None

    anchors = sorted(geometries[anchor],
                     key=lambda g: (-len(g[4]), g[5]))
    for v, r, w, h, ports, turns in anchors:
        if stopped[0]:
            break
        placements[anchor] = (v, r, 0, 0)
        pm = {(x, y): (anchor, v, k)
              for k, (x, y) in enumerate(ports)}
        dfs(tuple(i for i in range(count) if i != anchor),
            pm, len(ports), (0, 0, w, h), [], turns)
    placements[anchor] = None
    return [(sig, value[1]) for sig, value in
            sorted(results.items(), key=lambda kv: kv[1][0])[:300]]


def _click_point(m, i):
    o = m["objects"][i]
    own = _absolute(o)
    other = set()
    for j, q in enumerate(m["objects"]):
        if j != i:
            other.update(_absolute(q))
    bad = m.setdefault("badclick", set())
    ports = {(o["x"] + x, o["y"] + y) for x, y in _geom(o)[3]}
    candidates = [p for p in own if (i, p[0], p[1]) not in bad]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (
        p in other, p in ports,
        -sum(q in own for q in ((p[0] - 1, p[1]),
                               (p[0] + 1, p[1]),
                               (p[0], p[1] - 1),
                               (p[0], p[1] + 1))),
        p[1], p[0]))
    return candidates[0], candidates[0] not in other


def _emit_click(m, x, y, target=None, discover=False):
    s, off = m["scale"], m["off"]
    m["pending"] = {
        "kind": "discover" if discover else "select",
        "i": target, "point": (x, y),
    }
    return "ACTION6", off + y * s + s // 2, off + x * s + s // 2


def _emit_simple(m, action, probe=False):
    i = m.get("selected")
    if i is not None:
        o = m["objects"][i]
        n = m["n"]
        if action == "ACTION5":
            before = _geom(o)
            if probe:
                m["pending"] = {"kind": "probe", "before": before}
            elif o["family"]:
                o["v"], m["direction"] = _family_next(
                    o["v"], m["direction"], len(o["variants"]))
            else:
                m["pending"] = {"kind": "rotate", "i": i, "before": before}
                o["r"] = (o["r"] + 1) % 4
            if not probe:
                w, h, _, _ = _geom(o)
                o["x"] = max(0, min(o["x"], n - w))
                o["y"] = max(0, min(o["y"], n - h))
        else:
            dx, dy = {
                "ACTION1": (0, -1), "ACTION2": (0, 1),
                "ACTION3": (-1, 0), "ACTION4": (1, 0),
            }.get(action, (0, 0))
            w, h, _, _ = _geom(o)
            x, y = o["x"] + dx, o["y"] + dy
            if 0 <= x <= n - w and 0 <= y <= n - h:
                o["x"], o["y"] = x, y
    return action, None, None


def _process_pending(g, m):
    p = m.pop("pending", None)
    if p is None:
        return
    kind = p["kind"]
    sh = _capture(g, m)

    if kind == "discover":
        if sh is None:
            m["selected"] = None
            return
        i = _identify(g, m, sh)
        if i is None:
            i = len(m["objects"])
            m["objects"].append(_new_object(sh))
            if sh["special"] or len(sh["ports"]) >= 5:
                _begin_probe(m, i)
        m["selected"] = i

    elif kind == "select":
        actual = _identify(g, m, sh)
        m["selected"] = actual
        if actual != p["i"]:
            x, y = p["point"]
            m.setdefault("badclick", set()).add((p["i"], x, y))
        else:
            m.pop("defer", None)

    elif kind == "rotate":
        i = p["i"]
        o = m["objects"][i]
        if sh is not None and not _same_capture(sh, _geom(o)):
            # ACTION5 changed the shape rather than rotating it.
            w, h, mask, ports = p["before"]
            old = {"w": w, "h": h, "mask": tuple(sorted(mask)),
                   "ports": tuple(sorted(ports)),
                   "x": o["x"], "y": o["y"], "special": True}
            o["variants"] = [old, sh]
            o["v"], o["r"], o["family"] = 1, 0, True
            o["x"], o["y"] = sh["x"], sh["y"]
            m["selected"] = i
            m["phase"] = "probe"
            m["probe"] = {
                "i": i, "path": [0, 1], "edges": {(0, 1)}, "count": 1
            }
            m.pop("target", None)
            m.pop("solutions", None)
            m["failed"] = set()

    elif kind == "probe":
        if sh is None:
            m["phase"] = "discover"
            m.pop("probe", None)
            return
        pr = m["probe"]
        i = pr["i"]
        o = m["objects"][i]
        before = p["before"]
        w, h, mask, ports = before
        rotated = (h, w,
                   tuple((h - 1 - y, x) for x, y in mask),
                   tuple((h - 1 - y, x) for x, y in ports))
        if pr["count"] == 0 and _same_capture(sh, rotated):
            o["r"] = (o["r"] + 1) % 4
            o["x"], o["y"] = sh["x"], sh["y"]
            m["phase"] = "discover"
            m.pop("probe", None)
            return

        keys = [_shape_key(v) for v in o["variants"]]
        key = _shape_key(sh)
        if key in keys:
            v = keys.index(key)
        else:
            v = len(keys)
            o["variants"].append(sh)
        previous = pr["path"][-1]
        edge = (previous, v)
        repeated = edge in pr["edges"]
        pr["edges"].add(edge)
        pr["path"].append(v)
        pr["count"] += 1
        o["v"], o["r"], o["family"] = v, 0, True
        o["x"], o["y"] = sh["x"], sh["y"]
        m["selected"] = i
        if repeated or pr["count"] >= 22:
            _finish_probe(m)


def _choose_solution(m):
    if "solutions" not in m:
        m["solutions"] = _solve(m)
    best = None
    for sig, config in m["solutions"]:
        if sig in m["failed"]:
            continue
        scored = _shift_and_cost(config, m)
        if scored is not None:
            cost, target = scored
            if best is None or cost < best[0]:
                best = cost, sig, target
    if best is None:
        return False
    _, sig, target = best
    m["target"] = {"signature": sig, "poses": target, "touched": False}
    m["badclick"] = set()
    m.pop("defer", None)
    return True


def _movement_to(o, tx, ty):
    if o["y"] > ty:
        return "ACTION1"
    if o["y"] < ty:
        return "ACTION2"
    if o["x"] > tx:
        return "ACTION3"
    if o["x"] < tx:
        return "ACTION4"
    return None


def _free_step(m, i):
    o = m["objects"][i]
    w, h, _, _ = _geom(o)
    n = m["n"]
    for action, dx, dy in (
        ("ACTION1", 0, -1), ("ACTION2", 0, 1),
        ("ACTION3", -1, 0), ("ACTION4", 1, 0)
    ):
        if 0 <= o["x"] + dx <= n - w and 0 <= o["y"] + dy <= n - h:
            return action
    return "ACTION1"


def _execute(m):
    target = m["target"]
    objects = m["objects"]

    if m.get("check"):
        action = m["check"].pop(0)
        return _emit_simple(m, action)

    differing = []
    for i, (v, r, x, y) in enumerate(target["poses"]):
        o = objects[i]
        if (o["v"], o["r"], o["x"], o["y"]) != (v, r, x, y):
            differing.append(i)

    if not differing:
        if target["touched"]:
            m["failed"].add(target["signature"])
            m.pop("target", None)
            m.pop("check", None)
            return None
        i = m.get("selected")
        if i is None:
            for j in range(len(objects)):
                cp = _click_point(m, j)
                if cp:
                    return _emit_click(m, *cp[0], target=j)
            return None
        action = _free_step(m, i)
        opposite = {
            "ACTION1": "ACTION2", "ACTION2": "ACTION1",
            "ACTION3": "ACTION4", "ACTION4": "ACTION3",
        }[action]
        target["touched"] = True
        m["check"] = [opposite]
        return _emit_simple(m, action)

    selected = m.get("selected")
    deferred = m.get("defer")
    if selected in differing and (selected != deferred or len(differing) == 1):
        v, r, x, y = target["poses"][selected]
        o = objects[selected]
        if o["v"] != v or o["r"] != r:
            target["touched"] = True
            return _emit_simple(m, "ACTION5")
        action = _movement_to(o, x, y)
        if action:
            target["touched"] = True
            return _emit_simple(m, action)

    choices = []
    for i in differing:
        if i == deferred and len(differing) > 1:
            continue
        cp = _click_point(m, i)
        if cp is None:
            continue
        point, exclusive = cp
        v, r, x, y = target["poses"][i]
        o = objects[i]
        cost = (_turn_cost(o, v, r, m)
                + abs(x - o["x"]) + abs(y - o["y"]))
        choices.append((not exclusive, cost, i, point))
    if choices:
        _, _, i, point = min(choices)
        return _emit_click(m, *point, target=i)

    # Temporarily shift a covering selected shape to expose another one.
    if selected is not None:
        m["defer"] = selected
        target["touched"] = True
        return _emit_simple(m, _free_step(m, selected))
    return None


def act(obs, available, memory):
    obs = np.asarray(obs)
    if "_cn04" not in memory:
        m = {
            "objects": [], "selected": None, "phase": "discover",
            "direction": 1, "failed": set(), "badclick": set(),
            "bg": int(obs[63, 63]),
        }
        memory["_cn04"] = m
        g = _read_grid(obs, m)
        occupied = g != m["bg"]
        m["grey"] = bool(np.any(occupied & (g == 4)))
        sh = _capture(g, m)
        if sh is not None:
            m["objects"].append(_new_object(sh))
            m["selected"] = 0
            if sh["special"] or len(sh["ports"]) >= 5:
                _begin_probe(m, 0)
    else:
        m = memory["_cn04"]
        g = _read_grid(obs, m)
        _process_pending(g, m)

    if m["phase"] == "probe":
        return _emit_simple(m, "ACTION5", probe=True)

    if m["phase"] == "discover":
        covered = set()
        for o in m["objects"]:
            covered.update(_absolute(o))
        yy, xx = np.where(g != m["bg"])
        unknown = {(int(x), int(y)) for x, y in zip(xx, yy)
                   if (int(x), int(y)) not in covered}
        if unknown:
            point = max(
                unknown,
                key=lambda p: (
                    sum(q in unknown for q in (
                        (p[0] - 1, p[1]), (p[0] + 1, p[1]),
                        (p[0], p[1] - 1), (p[0], p[1] + 1))),
                    g[p[1], p[0]] not in (8, 3),
                    -p[1], -p[0]))
            return _emit_click(m, *point, discover=True)
        m["phase"] = "play"

    # A completed unsuccessful trial can immediately be replaced.
    for _ in range(3):
        if "target" not in m:
            if not _choose_solution(m):
                # Search again from the present, usually much closer, layout.
                m.pop("solutions", None)
                if not _choose_solution(m):
                    resets = memory.get("_cn04_resets", 0)
                    if "RESET" in available and resets < 2:
                        memory["_cn04_resets"] = resets + 1
                        memory.pop("_cn04", None)
                        return "RESET", None, None
                    i = m.get("selected")
                    if i is None and m["objects"]:
                        cp = _click_point(m, 0)
                        if cp:
                            return _emit_click(m, *cp[0], target=0)
                    m.pop("solutions", None)
                    return _emit_simple(m, "ACTION5")
        result = _execute(m)
        if result is not None:
            return result

    action = "ACTION5" if "ACTION5" in available else available[0]
    return action, None, None

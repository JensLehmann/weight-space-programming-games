import numpy as np
import math
import time
from collections import Counter


# These shapes supply type information when recognizable. Other shapes are
# discovered from observations and solved with initially unknown port types.
_TEMPLATE_ROWS = [
    ["ccc...", "c.ccc8", "c.....", "c.cccd", "ccc..."],
    [".eee", ".e.e", "8e.e", ".e.e", "de.e", ".e.e", ".eee"],
    ["db", ".b", ".b", ".b", ".b", ".d"],
    ["8.....d", "e.....e", "eee.eee", "..e.e..", "..eee.."],
    ["ffffff.", "f....fd", "f.ffff.", "f.f....", "f.f....",
     "fff....", ".8....."],
    [".b8.", "8b..", ".bbd", ".0.."],
    [".b8.", "8b..", ".bbd", "8b..", ".0.."],
    [".b8...", "8b....", ".bbd..", "8b....", ".bbbb8", ".0...."],
    [".b8...", "8b....", ".bbd..", "8b....", ".bbbb8",
     "8b....", ".0...."],
    [".b8...", "8b....", ".bbd..", "8b....", ".bbbb8",
     "8b....", ".bbbd."],
    ["8.8.8", "e.e.e", "eeeee"],
    ["8aaaa", "....a", ".daaa"],
    [".8c", "..c", "dcc"],
]


def _signature(p):
    return (
        p["w"], p["h"], tuple(sorted(p["c"])),
        tuple(sorted((x, y) for x, y, t in p["p"]))
    )


def _rotate(p):
    h = p["h"]
    q = {
        "w": h, "h": p["w"],
        "c": [(h - 1 - y, x) for x, y in p["c"]],
        "p": [(h - 1 - y, x, t) for x, y, t in p["p"]],
        "col": {(h - 1 - y, x): v for (x, y), v in p["col"].items()},
        "known": p.get("known"),
        "ori": (p.get("ori", 0) + 1) % 4,
    }
    if "fi" in p:
        q["fi"] = p["fi"]
    if "pid" in p:
        q["pid"] = {
            (h - 1 - y, x): v for (x, y), v in p["pid"].items()
        }
    q["sig"] = _signature(q)
    return q


def _build_library():
    library = {}
    families = []
    for i, rows in enumerate(_TEMPLATE_ROWS):
        p = {
            "w": len(rows[0]), "h": len(rows),
            "c": [], "p": [], "col": {},
            "ori": 0,
            "known": "family" if 5 <= i <= 9 else "rot",
        }
        if 5 <= i <= 9:
            p["fi"] = i - 5
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch != ".":
                    value = int(ch, 16)
                    p["c"].append((x, y))
                    p["col"][(x, y)] = value
                    if value in (8, 13):
                        p["p"].append((x, y, value))
        p["sig"] = _signature(p)
        if 5 <= i <= 9:
            families.append(p)
        for _ in range(4):
            library.setdefault(p["sig"], p)
            p = _rotate(p)
    return library, families


_LIBRARY, _FAMILY = _build_library()
_LIBRARY_VALUES = list(_LIBRARY.values())


def _components(mask):
    h, w = mask.shape
    seen = set()
    result = []
    for yy, xx in zip(*np.where(mask)):
        start = (int(xx), int(yy))
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        component = []
        while stack:
            x, y = stack.pop()
            component.append((x, y))
            for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if (0 <= nx < w and 0 <= ny < h
                        and mask[ny, nx] and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    stack.append((nx, ny))
        result.append(component)
    return result


def _initialize(obs, m):
    resets = m.get("resets", 0)
    bg = int(obs[-1, -1])
    runs = []
    for y in range(2, 63):
        row = obs[y]
        j = 0
        while j < 64:
            k = j + 1
            while k < 64 and row[k] == row[j]:
                k += 1
            if row[j] != bg and k - j < 40:
                runs.append(k - j)
            j = k
    scale = 0
    for r in runs:
        scale = math.gcd(scale, r)
    if scale not in (2, 3, 4, 5, 6):
        scale = 3
    if scale == 3:
        size, offset = 20, 2
    else:
        size = 64 // scale
        offset = (64 - size * scale) // 2
    m.clear()
    m.update({
        "scale": scale, "offset": offset,
        "W": size, "H": size, "bg": bg,
        "objects": [], "sel": None, "phase": "discover",
        "wait": None, "direction": 1, "failed": set(),
        "plan": None, "touched": False, "discover_steps": 0,
        "untyped": False, "misses": 0, "badclick": 0,
        "resets": resets,
    })


def _grid(obs, m):
    s, o = m["scale"], m["offset"]
    return np.asarray(obs)[
        o + s//2:o + s*m["H"]:s,
        o + s//2:o + s*m["W"]:s
    ]


def _expected_color(value, grey):
    if value in (8, 13):
        return 8
    if grey:
        return value
    return 3 if value == 3 else 0


def _selected_at(g, m, p, x, y):
    if x < 0 or y < 0 or x + p["w"] > m["W"] or y + p["h"] > m["H"]:
        return False
    body = False
    ports = {(a, b) for a, b, t in p["p"]}
    for a, b in p["c"]:
        actual = int(g[y+b, x+a])
        if (a, b) in ports:
            if actual not in (8, 3):
                return False
        else:
            wanted = _expected_color(p["col"].get((a, b), 0), m["grey"])
            if actual != wanted:
                return False
            body = True
    return body


def _known_capture(g, m):
    dots = [(int(x), int(y)) for y, x in
            zip(*np.where((g == 8) | (g == 3)))]
    if not dots:
        return None
    if m["grey"]:
        bodymask = ((g != m["bg"]) & (g != 4) &
                    (g != 8) & (g != 3))
    else:
        bodymask = g == 0
    body = {(int(x), int(y)) for y, x in zip(*np.where(bodymask))}
    if not body:
        return None

    best = None
    bestscore = None
    for p in _LIBRARY_VALUES:
        if not p["p"]:
            continue
        ax, ay, _ = p["p"][0]
        for dx, dy in dots:
            x, y = dx-ax, dy-ay
            if not _selected_at(g, m, p, x, y):
                continue
            cells = {(x+a, y+b) for a, b in p["c"]}
            outside = len(body - cells)
            if outside > max(1, len(body)//12):
                continue
            score = (-outside, len(p["c"]))
            if bestscore is None or score > bestscore:
                bestscore = score
                best = dict(p)
                best["x"], best["y"] = x, y
    return best


def _capture_cells(cells, g):
    if not cells:
        return None
    x0 = min(x for x, y in cells)
    y0 = min(y for x, y in cells)
    x1 = max(x for x, y in cells)
    y1 = max(y for x, y in cells)
    p = {
        "x": x0, "y": y0, "w": x1-x0+1, "h": y1-y0+1,
        "c": [(x-x0, y-y0) for x, y in cells],
        "p": [(x-x0, y-y0, -1) for x, y in cells
              if int(g[y, x]) in (8, 3)],
        "col": {(x-x0, y-y0): int(g[y, x]) for x, y in cells},
        "known": None, "ori": 0,
    }
    p["sig"] = _signature(p)
    known = _LIBRARY.get(p["sig"])
    if known is not None:
        q = dict(known)
        q["x"], q["y"] = x0, y0
        return q
    return p


def _capture(g, m, hint=None):
    known = _known_capture(g, m)
    if known is not None:
        return known

    if m["grey"]:
        bodymask = ((g != m["bg"]) & (g != 4) &
                    (g != 8) & (g != 3))
        mask = (g != m["bg"]) & (g != 4)
    else:
        bodymask = g == 0
        mask = (g == 0) | (g == 8) | (g == 3)
    if not np.any(bodymask):
        return None

    if hint:
        valid = True
        hasbody = False
        for x, y in hint:
            if not (0 <= x < m["W"] and 0 <= y < m["H"]):
                valid = False
                break
            v = int(g[y, x])
            if (m["grey"] and v == 4) or (
                    not m["grey"] and v not in (0, 8, 3)):
                valid = False
                break
            hasbody |= bool(bodymask[y, x])
        if valid and hasbody:
            return _capture_cells(hint, g)

    comps = _components(mask)
    comps = [c for c in comps if any(bodymask[y, x] for x, y in c)]
    if not comps:
        return None
    cells = max(comps, key=lambda c: sum(bodymask[y, x] for x, y in c))
    return _capture_cells(cells, g)


def _rot_object(cap, q=0):
    cap = dict(cap)
    cap["pid"] = {(x, y): k for k, (x, y, t) in enumerate(cap["p"])}
    opts = [cap]
    for _ in range(3):
        opts.append(_rotate(opts[-1]))
    return {
        "kind": "rot", "opts": opts, "q": q,
        "x": cap["x"], "y": cap["y"], "done": True,
    }


def _set_family_ports(opts):
    result = []
    for q, p in enumerate(opts):
        p = dict(p)
        p["pid"] = {
            (x, y): (q+1)*4096 + y*64 + x for x, y, t in p["p"]
        }
        result.append(p)
    return result


def _known_family(obj, old, cap, m):
    if (old.get("known") != "family" or cap.get("known") != "family"
            or old.get("ori") != cap.get("ori")):
        return False
    opts = []
    for base in _FAMILY:
        p = base
        for _ in range(old.get("ori", 0)):
            p = _rotate(p)
        opts.append(p)
    matches = [i for i, p in enumerate(opts) if p["sig"] == cap["sig"]]
    if not matches:
        return False
    q = matches[0]
    oldq = old["fi"]
    obj.clear()
    obj.update({
        "kind": "family", "done": True,
        "opts": _set_family_ports(opts), "q": q,
        "x": cap["x"], "y": cap["y"],
    })
    if q != oldq:
        m["direction"] = 1 if q > oldq else -1
    return True


def _add_object(cap, m):
    if cap.get("known") == "rot":
        obj = _rot_object(cap)
    else:
        obj = {
            "kind": "probe", "raw": cap, "history": [cap],
            "x": cap["x"], "y": cap["y"], "q": 0, "done": False,
            "tries": 0,
        }
    m["objects"].append(obj)
    m["sel"] = len(m["objects"]) - 1


def _current_shape(obj):
    return obj["opts"][obj["q"]] if obj.get("done") else obj["raw"]


def _finish_family(obj, m):
    history = obj["history"]
    unique, adjacency = {}, {}
    for p in history:
        unique.setdefault(p["sig"], p)
        adjacency.setdefault(p["sig"], set())
    for a, b in zip(history, history[1:]):
        if a["sig"] != b["sig"]:
            adjacency[a["sig"]].add(b["sig"])
            adjacency[b["sig"]].add(a["sig"])
    ends = [s for s in unique if len(adjacency[s]) <= 1]
    start = min(ends or list(unique),
                key=lambda s: (len(unique[s]["c"]), unique[s]["h"], s))
    order = []
    previous, here = None, start
    while here not in order:
        order.append(here)
        nxt = [v for v in adjacency[here]
               if v != previous and v not in order]
        if not nxt:
            break
        previous, here = here, nxt[0]
    order.extend(s for s in unique if s not in order)
    q = order.index(history[-1]["sig"])
    if len(history) >= 2:
        oldq = order.index(history[-2]["sig"])
        if oldq != q:
            m["direction"] = 1 if q > oldq else -1
    opts = _set_family_ports([unique[s] for s in order])
    x, y = history[-1]["x"], history[-1]["y"]
    obj.clear()
    obj.update({
        "kind": "family", "done": True, "opts": opts,
        "q": q, "x": x, "y": y,
    })


def _click_action(x, y, m, wait):
    m["wait"] = wait
    s, o = m["scale"], m["offset"]
    return ("ACTION6", int(o+y*s+s//2), int(o+x*s+s//2))


def _simple_action(n, m, wait):
    m["wait"] = wait
    return ("ACTION" + str(n), None, None)


def _advance_variant(obj, direction):
    q = obj["q"]
    if obj["kind"] == "rot":
        return (q+1) % 4, direction
    nq = q + direction
    if not 0 <= nq < len(obj["opts"]):
        direction = -direction
        nq = q + direction
    return max(0, min(len(obj["opts"])-1, nq)), direction


def _update(g, m):
    wait = m.get("wait")
    m["wait"] = None
    if wait is None:
        return
    kind = wait[0]
    objects = m["objects"]

    if kind == "new":
        cap = _capture(g, m, wait[1])
        if cap is None:
            m["sel"] = None
            m["misses"] += 1
            return
        for i, obj in enumerate(objects):
            if (_current_shape(obj)["sig"] == cap["sig"]
                    and obj["x"] == cap["x"] and obj["y"] == cap["y"]):
                m["sel"] = i
                m["misses"] += 1
                return
        _add_object(cap, m)
        m["misses"] = 0
        return

    if kind in ("probe", "family"):
        i = wait[1]
        obj = objects[i]
        cap = _capture(g, m)
        if cap is None:
            m["sel"] = None
            m["misses"] += 1
            return
        m["sel"] = i
        old = obj["raw"]
        if kind == "probe":
            if cap["sig"] == _rotate(old)["sig"]:
                new = _rot_object(old, 1)
                new["x"], new["y"] = cap["x"], cap["y"]
                objects[i] = new
            elif cap["sig"] == old["sig"]:
                obj["tries"] = obj.get("tries", 0) + 1
                m["sel"] = None
                m["misses"] += 1 + (2 if obj["tries"] >= 2 else 0)
                return
            elif _known_family(obj, old, cap, m):
                pass
            else:
                obj["kind"] = "family"
                obj["history"].append(cap)
                obj["raw"] = cap
                obj["x"], obj["y"] = cap["x"], cap["y"]
        else:
            obj["history"].append(cap)
            obj["raw"] = cap
            obj["x"], obj["y"] = cap["x"], cap["y"]
            h = obj["history"]
            repeat = (len(h) >= 4 and h[-2]["sig"] == h[0]["sig"]
                      and h[-1]["sig"] == h[1]["sig"])
            if repeat or len(h) >= 19:
                _finish_family(obj, m)
        m["misses"] = 0
        return

    if kind == "select":
        wanted = wait[1]
        obj = objects[wanted]
        if _selected_at(g, m, _current_shape(obj), obj["x"], obj["y"]):
            m["sel"] = wanted
            m["misses"] = 0
            return
        for i, obj in enumerate(objects):
            if _selected_at(g, m, _current_shape(obj), obj["x"], obj["y"]):
                m["sel"] = i
                m["misses"] += 1
                return
        m["sel"] = None
        m["misses"] += 1
        return

    if kind == "do":
        i, action = wait[1], wait[2]
        obj = objects[i]
        oldq = obj["q"]
        x, y = obj["x"], obj["y"]
        q = oldq
        if action == 5:
            q, direction = _advance_variant(obj, m["direction"])
            if obj["kind"] == "family":
                cap = _capture(g, m)
                if cap is not None:
                    matches = [j for j, p in enumerate(obj["opts"])
                               if p["sig"] == cap["sig"]]
                    if matches:
                        q = min(matches, key=lambda j: abs(j-q))
                if q != oldq:
                    direction = 1 if q > oldq else -1
                m["direction"] = direction
            p = obj["opts"][q]
            x = max(0, min(x, m["W"]-p["w"]))
            y = max(0, min(y, m["H"]-p["h"]))
        else:
            dx, dy = {1: (0, -1), 2: (0, 1),
                      3: (-1, 0), 4: (1, 0)}[action]
            p = obj["opts"][q]
            nx, ny = x+dx, y+dy
            if (nx >= 0 and ny >= 0 and
                    nx+p["w"] <= m["W"] and ny+p["h"] <= m["H"]):
                x, y = nx, ny

        # Crucially, the bounding box is updated by the transition rules,
        # not by a component that might include another piece's terminals.
        obj["q"], obj["x"], obj["y"] = q, x, y
        m["sel"] = i
        m["touched"] = True
        if _selected_at(g, m, obj["opts"][q], x, y):
            m["misses"] = 0
        else:
            m["misses"] += 1


def _option_costs(obj, m):
    result = []
    for target, p in enumerate(obj["opts"]):
        q = obj["q"]
        x, y = obj["x"], obj["y"]
        direction = m["direction"]
        steps = 0
        while q != target and steps < 4*len(obj["opts"])+4:
            if obj["kind"] == "rot":
                q = (q+1) % 4
            else:
                nq = q+direction
                if not 0 <= nq < len(obj["opts"]):
                    direction = -direction
                    nq = q+direction
                q = max(0, min(len(obj["opts"])-1, nq))
            shape = obj["opts"][q]
            x = max(0, min(x, m["W"]-shape["w"]))
            y = max(0, min(y, m["H"]-shape["h"]))
            steps += 1
        ports = [
            (a, b, -1 if m["untyped"] else t, p["pid"][(a, b)])
            for a, b, t in p["p"]
        ]
        result.append({
            "q": target, "w": p["w"], "h": p["h"], "p": ports,
            "turn": steps, "ax": x, "ay": y,
        })
    return result


def _search(m, seconds):
    objects = m["objects"]
    n = len(objects)
    if not n or n > 11 or seconds <= 0:
        return None
    deadline = time.perf_counter() + seconds
    options = [_option_costs(o, m) for o in objects]
    W, H = m["W"], m["H"]
    full = (1 << n)-1
    maxports = [max(len(p["p"]) for p in op) for op in options]
    capacities = [0]*(1 << n)
    parities = [set() for _ in range(1 << n)]
    parities[0] = {0}
    for mask in range(1, 1 << n):
        bit = mask & -mask
        i = bit.bit_length()-1
        rest = mask ^ bit
        capacities[mask] = capacities[rest]+maxports[i]
        parities[mask] = {
            a ^ (len(p["p"]) & 1)
            for a in parities[rest] for p in options[i]
        }

    best = [float("inf"), None]
    nodes = [0]

    def translation(placed):
        lx = max(-x for i, p, x, y in placed)
        hx = min(W-p["w"]-x for i, p, x, y in placed)
        ly = max(-y for i, p, x, y in placed)
        hy = min(H-p["h"]-y for i, p, x, y in placed)
        if lx > hx or ly > hy:
            return None
        vx = sorted(p["ax"]-x for i, p, x, y in placed)
        vy = sorted(p["ay"]-y for i, p, x, y in placed)
        k = len(placed)
        lo_x = min(hx, max(lx, vx[(k-1)//2]))
        hi_x = min(hx, max(lx, vx[k//2]))
        lo_y = min(hy, max(ly, vy[(k-1)//2]))
        hi_y = min(hy, max(ly, vy[k//2]))
        mx, my = hi_x, hi_y
        for i, p, x, y in placed:
            if i == m["sel"]:
                mx = min(hi_x, max(lo_x, p["ax"]-x))
                my = min(hi_y, max(lo_y, p["ay"]-y))
                break
        cost = sum(
            p["turn"] + abs(p["ax"]-x-mx) + abs(p["ay"]-y-my)
            for i, p, x, y in placed
        )
        return cost, mx, my

    def save(placed, occupancy, tr):
        # Port identities remain fixed under rotation. Thus failed pairings
        # are excluded even when translated or globally rotated.
        key = tuple(sorted(
            tuple(sorted(v[2])) for v in occupancy.values()
        ))
        if key in m["failed"]:
            return
        cost, tx, ty = tr
        target = [None]*n
        for i, p, x, y in placed:
            xx, yy = x+tx, y+ty
            target[i] = (p["q"], xx, yy)
            o = objects[i]
            changed = (o["q"], o["x"], o["y"]) != (p["q"], xx, yy)
            if changed and i != m["sel"]:
                cost += 1
        if cost < best[0]:
            best[:] = [cost, {"target": target, "key": key}]

    def recurse(placed, remaining, occupancy, tr):
        nodes[0] += 1
        if nodes[0] % 96 == 0 and time.perf_counter() >= deadline:
            raise TimeoutError
        if tr is None or tr[0] > best[0]:
            return
        singles = [(xy, v[0]) for xy, v in occupancy.items() if v[1] == 1]
        if not remaining:
            if not singles:
                save(placed, occupancy, tr)
            return
        if len(singles) > capacities[remaining]:
            return
        if (len(singles) & 1) not in parities[remaining]:
            return

        if singles:
            def rarity(item):
                t = item[1]
                if t < 0:
                    return 100000
                return sum(
                    sum(tt < 0 or tt == t for xx, yy, tt, pid in p["p"])
                    for i in range(n) if remaining & (1 << i)
                    for p in options[i]
                )
            point, terminal_type = min(singles, key=rarity)
            ids = [i for i in range(n) if remaining & (1 << i)]
        else:
            ids = [(remaining & -remaining).bit_length()-1]
            point, terminal_type = None, -1

        branches = []
        for i in ids:
            for p in options[i]:
                if point is None:
                    root_i, root_p, root_x, root_y = placed[0]
                    offsets = [
                        (objects[i]["x"]-objects[root_i]["x"]+root_x,
                         objects[i]["y"]-objects[root_i]["y"]+root_y)
                    ]
                else:
                    offsets = [
                        (point[0]-px, point[1]-py)
                        for px, py, t, pid in p["p"]
                        if terminal_type < 0 or t < 0 or terminal_type == t
                    ]
                seen_offsets = set()
                for x, y in offsets:
                    if (x, y) in seen_offsets:
                        continue
                    seen_offsets.add((x, y))
                    valid = True
                    additions = []
                    for px, py, t, pid in p["p"]:
                        xy = (x+px, y+py)
                        old = occupancy.get(xy)
                        identity = (i, pid)
                        if old is not None:
                            if (old[1] == 2 or
                                    (old[0] >= 0 and t >= 0 and old[0] != t)):
                                valid = False
                                break
                            additions.append(
                                (xy, (old[0], 2, old[2]+(identity,)))
                            )
                        else:
                            additions.append((xy, (t, 1, (identity,))))
                    if not valid:
                        continue
                    newplaced = placed + [(i, p, x, y)]
                    nt = translation(newplaced)
                    if nt is None or nt[0] > best[0]:
                        continue
                    newocc = occupancy.copy()
                    newocc.update(additions)
                    left = remaining ^ (1 << i)
                    ns = sum(v[1] == 1 for v in newocc.values())
                    if ns > capacities[left] or (ns & 1) not in parities[left]:
                        continue
                    branches.append((nt[0], newplaced, left, newocc, nt))
        branches.sort(key=lambda a: a[0])
        for _, newplaced, left, newocc, nt in branches:
            recurse(newplaced, left, newocc, nt)

    root = max(range(n), key=lambda i: (maxports[i], i == m["sel"]))
    roots = sorted(options[root], key=lambda p: (p["turn"], -len(p["p"])))
    try:
        for p in roots:
            placed = [(root, p, 0, 0)]
            occ = {
                (x, y): (t, 1, ((root, pid),))
                for x, y, t, pid in p["p"]
            }
            recurse(placed, full ^ (1 << root), occ, translation(placed))
            if time.perf_counter() >= deadline:
                break
    except TimeoutError:
        pass
    return best[1]


def _click_piece(i, m):
    objects = m["objects"]
    obj = objects[i]
    p = _current_shape(obj)
    occupied = Counter()
    for j, other in enumerate(objects):
        if i == j:
            continue
        for x, y in _current_shape(other)["c"]:
            occupied[(other["x"]+x, other["y"]+y)] += 1
    ports = {(x, y) for x, y, t in p["p"]}
    choices = []
    for x, y in p["c"]:
        wx, wy = obj["x"]+x, obj["y"]+y
        if 0 <= wx < m["W"] and 0 <= wy < m["H"]:
            choices.append((occupied[(wx, wy)], (x, y) in ports, wy, wx))
    if not choices:
        return None
    _, _, y, x = min(choices)
    return _click_action(x, y, m, ("select", i))


def _free_move(obj, m):
    p = _current_shape(obj)
    choices = [
        (4, obj["x"]+p["w"] < m["W"]),
        (2, obj["y"]+p["h"] < m["H"]),
        (3, obj["x"] > 0),
        (1, obj["y"] > 0),
    ]
    return next((a for a, ok in choices if ok), 5)


def _reset(m, available):
    count = m.get("resets", 0)+1
    if "RESET" in available and count <= 2:
        m.clear()
        m["resets"] = count
        return ("RESET", None, None)
    for a in ("ACTION5", "ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        if a in available:
            return (a, None, None)
    return ("ACTION6", 32, 32)


def act(obs, available, memory):
    deadline = time.perf_counter()+7.8
    m = memory
    if "objects" not in m:
        _initialize(np.asarray(obs), m)
        g = _grid(obs, m)
        m["grey"] = bool(np.any(g == 4) and m["bg"] != 4)
        cap = _capture(g, m)
        if cap is not None:
            _add_object(cap, m)
    else:
        g = _grid(obs, m)
        wait = m.get("wait")
        if (wait and wait[0] in ("select", "new") and
                "lastgrid" in m and np.array_equal(g, m["lastgrid"])):
            m["badclick"] += 1
        else:
            m["badclick"] = 0
        _update(g, m)
    m["lastgrid"] = g.copy()

    if m["badclick"] >= 3 or m["misses"] >= 3:
        return _reset(m, available)

    if m["phase"] == "discover":
        m["discover_steps"] += 1
        if m["discover_steps"] > 65 or len(m["objects"]) > 11:
            return _reset(m, available)

        for i, obj in enumerate(m["objects"]):
            if not obj.get("done"):
                if m["sel"] != i:
                    return _click_piece(i, m) or _reset(m, available)
                kind = "probe" if obj["kind"] == "probe" else "family"
                return _simple_action(5, m, (kind, i))

        remaining = (g != m["bg"]).copy()
        for obj in m["objects"]:
            for x, y in _current_shape(obj)["c"]:
                xx, yy = obj["x"]+x, obj["y"]+y
                if 0 <= xx < m["W"] and 0 <= yy < m["H"]:
                    remaining[yy, xx] = False

        if np.any(remaining):
            comps = _components(remaining)
            comp = max(comps, key=len)
            # Prefer interior/body pixels over possibly shared terminals.
            choices = []
            cellset = set(comp)
            for x, y in comp:
                degree = sum((x+dx, y+dy) in cellset
                             for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                choices.append((int(g[y, x]) in (8, 3), -degree, y, x))
            _, _, y, x = min(choices)
            return _click_action(x, y, m, ("new", comp))

        if not m["objects"]:
            return _reset(m, available)
        m["phase"] = "play"
        m["plan"] = None
        m["touched"] = False

    for _ in range(3):
        if m["plan"] is None:
            budget = min(3.5, deadline-time.perf_counter())
            m["plan"] = _search(m, budget)
            if m["plan"] is None and not m["untyped"]:
                m["untyped"] = True
                budget = min(2.8, deadline-time.perf_counter())
                m["plan"] = _search(m, budget)
            if m["plan"] is None:
                return _reset(m, available)
            m["touched"] = False

        targets = m["plan"]["target"]
        unfinished = [
            i for i, target in enumerate(targets)
            if (m["objects"][i]["q"], m["objects"][i]["x"],
                m["objects"][i]["y"]) != target
        ]

        if not unfinished:
            if not m["touched"]:
                # Selection itself does not perform the victory test.
                i = m["sel"]
                if i is None:
                    return _click_piece(0, m) or _reset(m, available)
                a = _free_move(m["objects"][i], m)
                return _simple_action(a, m, ("do", i, a))
            m["failed"].add(m["plan"]["key"])
            m["plan"] = None
            continue

        i = m["sel"] if m["sel"] in unfinished else unfinished[0]
        if m["sel"] != i:
            if m["misses"] >= 1 and m["sel"] is not None:
                # Move an occluding/click-intercepting piece temporarily.
                j = m["sel"]
                a = _free_move(m["objects"][j], m)
                m["misses"] = 0
                return _simple_action(a, m, ("do", j, a))
            return _click_piece(i, m) or _reset(m, available)

        obj = m["objects"][i]
        q, x, y = targets[i]
        if obj["q"] != q:
            return _simple_action(5, m, ("do", i, 5))
        if obj["x"] < x:
            a = 4
        elif obj["x"] > x:
            a = 3
        elif obj["y"] < y:
            a = 2
        else:
            a = 1
        return _simple_action(a, m, ("do", i, a))

    return _reset(m, available)

import math
import time
import numpy as np


_KNOWN_TEXT = [
    ["###...", "#.###a", "#.....", "#.###b", "###..."],
    [".###", ".#.#", "a#.#", ".#.#", "b#.#", ".#.#", ".###"],
    ["b#", ".#", ".#", ".#", ".#", ".b"],
    ["a.....b", "#.....#", "###.###", "..#.#..", "..###.."],
    ["######.", "#....#b", "#.####.", "#.#....",
     "#.#....", "###....", ".a....."],
    [".#a.", "a#..", ".##b", ".z.."],
    [".#a.", "a#..", ".##b", "a#..", ".z.."],
    [".#a...", "a#....", ".##b..", "a#....", ".####a", ".z...."],
    [".#a...", "a#....", ".##b..", "a#....",
     ".####a", "a#....", ".z...."],
    [".#a...", "a#....", ".##b..", "a#....",
     ".####a", "a#....", ".###b."],
    ["a.a.a", "#.#.#", "#####"],
    ["a####", "....#", ".b###"],
    [".a#", "..#", "b##"],
]


def _rot(points, w, h, r):
    if r == 0:
        return list(points)
    if r == 1:
        return [(h - 1 - y, x) for x, y in points]
    if r == 2:
        return [(w - 1 - x, h - 1 - y) for x, y in points]
    return [(y, w - 1 - x) for x, y in points]


def _shape(cells, ports):
    cells = sorted(set(cells), key=lambda q: (q[1], q[0]))
    ports = sorted(set(ports), key=lambda q: (q[1], q[0]))
    w = max(x for x, y in cells) + 1
    h = max(y for x, y in cells) + 1
    result = {
        "cells": cells, "ports": ports, "labels": [-1] * len(ports),
        "w": w, "h": h, "hint": False,
        "final_hint": False, "recognized": False,
    }
    cs, ps = set(cells), set(ports)
    for k, text in enumerate(_KNOWN_TEXT):
        tw, th = len(text[0]), len(text)
        tc, tp, labels = [], [], []
        for y, line in enumerate(text):
            for x, char in enumerate(line):
                if char != ".":
                    tc.append((x, y))
                if char in "ab":
                    tp.append((x, y))
                    labels.append(0 if char == "a" else 1)
        for r in range(4):
            rw, rh = (tw, th) if r % 2 == 0 else (th, tw)
            if (rw, rh) != (w, h):
                continue
            rp = _rot(tp, tw, th, r)
            if set(_rot(tc, tw, th, r)) == cs and set(rp) == ps:
                lut = dict(zip(rp, labels))
                result.update(
                    labels=[lut[q] for q in ports],
                    hint=5 <= k <= 9,
                    final_hint=k == 9,
                    recognized=True,
                )
                return result
    return result


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            seen[y, x] = True
            stack, cells = [(x, y)], []
            while stack:
                xx, yy = stack.pop()
                cells.append((xx, yy))
                for nx, ny in ((xx - 1, yy), (xx + 1, yy),
                               (xx, yy - 1), (xx, yy + 1)):
                    if (0 <= nx < w and 0 <= ny < h
                            and mask[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            result.append(cells)
    return result


def _decode(obs, m):
    if "scale" not in m:
        bg = int(obs[-1, -1])
        lengths = []

        # Crucially, never include the HUD at display row zero.
        # Horizontal rows are complete; vertical lines start at row one.
        lines = [obs[y, :] for y in range(1, 64)]
        lines += [obs[1:, x] for x in range(64)]
        for line in lines:
            start = 0
            for j in range(1, len(line) + 1):
                if j == len(line) or line[j] != line[start]:
                    length = j - start
                    if int(line[start]) != bg and length < 45:
                        lengths.append(length)
                    start = j
        scale = 0
        for length in lengths:
            scale = math.gcd(scale, length)
        if scale < 1 or scale > 8:
            scale = 3
        m["scale"] = scale
        m["offset"] = (64 % scale) // 2
        m["bg"] = bg

    s, off = m["scale"], m["offset"]
    n = (64 - 2 * off) // s
    centers = off + np.arange(n) * s + s // 2
    g = obs[np.ix_(centers, centers)].astype(int)
    # For an unscaled 64x64 viewport, remove the HUD from segmentation.
    if s == 1:
        g[0, :] = m["bg"]
    m["W"], m["H"] = g.shape[1], g.shape[0]
    return g


def _option(p, v=None, r=None):
    v = p["v"] if v is None else v
    r = p["r"] if r is None else r
    sh = p["variants"][v]
    if p["hub"]:
        r = 0
    w, h = sh["w"], sh["h"]
    return {
        "v": v, "r": r,
        "cells": _rot(sh["cells"], w, h, r),
        "ports": _rot(sh["ports"], w, h, r),
        "labels": sh["labels"],
        "w": w if r % 2 == 0 else h,
        "h": h if r % 2 == 0 else w,
    }


def _active(g, m):
    best, score = None, 0
    for i, p in enumerate(m["pieces"]):
        total = 0
        for dx, dy in _option(p)["cells"]:
            x, y = p["x"] + dx, p["y"] + dy
            if 0 <= x < m["W"] and 0 <= y < m["H"]:
                c = int(g[y, x])
                if m["grey"]:
                    total += c not in (m["bg"], 4, 3)
                else:
                    total += c == 0
        if total > score:
            best, score = i, total
    return best


def _capture_existing(g, p):
    op = _option(p)
    ports, zeros = [], 0
    for dx, dy in op["cells"]:
        x, y = p["x"] + dx, p["y"] + dy
        if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
            c = int(g[y, x])
            if c in (8, 3):
                ports.append((dx, dy))
            zeros += c == 0
    sh = _shape(op["cells"], ports)
    p.update(variants=[sh], v=0, r=0, learned=True, zeros=zeros)
    if sh["hint"] and sh["final_hint"]:
        p["hub"], p["tested"] = True, True


def _capture_selected(g, m, p):
    if m["grey"]:
        mask = (g != m["bg"]) & (g != 4)
    else:
        mask = (g == 0) | (g == 8) | (g == 3)
    comps = _components(mask)
    oldset = {(p["x"] + x, p["y"] + y)
              for x, y in _option(p)["cells"]}
    if not comps:
        return None

    def rank(c):
        overlap = sum(q in oldset for q in c)
        distance = min(abs(x - p["x"]) + abs(y - p["y"]) for x, y in c)
        return overlap * 100 - distance + len(c) * 0.01

    cells = max(comps, key=rank)
    x0 = min(x for x, y in cells)
    y0 = min(y for x, y in cells)
    local = [(x - x0, y - y0) for x, y in cells]
    ports = [(x - x0, y - y0) for x, y in cells
             if int(g[y, x]) in (8, 3)]
    zeros = sum(int(g[y, x]) == 0 for x, y in cells)
    return x0, y0, _shape(local, ports), zeros


def _signature(sh):
    return tuple(sh["cells"]), tuple(sh["ports"])


def _needs_probe(p, grey):
    if p["tested"]:
        return False
    sh = p["variants"][0]
    if sh["hint"]:
        return not sh["final_hint"]
    if sh["recognized"]:
        return False
    return len(sh["ports"]) >= 3 or (grey and p.get("zeros", 0) == 1)


def _initialize(g, m):
    comps = [c for c in _components(g != m["bg"]) if len(c) >= 2]
    if not comps:
        return False
    grey = any(int(g[y, x]) == 4 for c in comps for x, y in c)
    m["grey"] = grey
    pieces = []
    for cells in comps:
        x0 = min(x for x, y in cells)
        y0 = min(y for x, y in cells)
        local = [(x - x0, y - y0) for x, y in cells]
        ports = [(x - x0, y - y0) for x, y in cells
                 if int(g[y, x]) in (8, 3)]
        sh = _shape(local, ports)
        p = {
            "x": x0, "y": y0, "variants": [sh],
            "r": 0, "v": 0, "hub": False,
            "learned": not grey, "tested": False, "zeros": 0,
        }
        if sh["hint"] and sh["final_hint"]:
            p["hub"], p["tested"] = True, True
        pieces.append(p)
    m.update(pieces=pieces, phase="learn", direction=1,
             failed=set(), lastcheck=False)
    m["active"] = _active(g, m)
    if m["active"] is not None:
        _capture_existing(g, pieces[m["active"]])
    return True


def _click(m, i):
    p = m["pieces"][i]
    op = _option(p)
    others = []
    for j, q in enumerate(m["pieces"]):
        if j != i:
            others.append({(q["x"] + x, q["y"] + y)
                           for x, y in _option(q)["cells"]})
    choices, ports = [], set(op["ports"])
    for dx, dy in op["cells"]:
        x, y = p["x"] + dx, p["y"] + dy
        if 0 <= x < m["W"] and 0 <= y < m["H"]:
            covering = sum((x, y) in cells for cells in others)
            choices.append((covering, (dx, dy) in ports, y, x))
    if not choices:
        return "ACTION5", None, None
    _, _, y, x = min(choices)
    s, off = m["scale"], m["offset"]
    return "ACTION6", off + y * s + s // 2, off + x * s + s // 2


def _hub_distance(p, target, direction):
    n, v, d = len(p["variants"]), p["v"], direction
    if n <= 1:
        return 0
    for steps in range(2 * n + 2):
        if v == target:
            return steps
        nv = v + d
        if nv >= n or nv < 0:
            d = -d
            nv = v + d
        v = nv
    return 2 * n


def _search(m):
    pieces = m["pieces"]
    n, W, H = len(pieces), m["W"], m["H"]
    options = []
    for p in pieces:
        if p["hub"]:
            oo = [_option(p, v, 0) for v in range(len(p["variants"]))]
        else:
            oo = [_option(p, 0, r) for r in range(4)]
        oo = [o for o in oo if o["ports"] and o["w"] <= W and o["h"] <= H]
        oo.sort(key=lambda o: (
            _hub_distance(p, o["v"], m["direction"])
            if p["hub"] else (o["r"] - p["r"]) % 4))
        options.append(oo)
    if not n or any(not oo for oo in options):
        return None

    anchor = max(range(n), key=lambda i: max(len(o["ports"]) for o in options[i]))
    deadline = time.monotonic() + 3.0
    visits = 0
    answers, seen = {}, set()

    def bounds(placed):
        return (
            min(t[1] for t in placed.values()),
            min(t[2] for t in placed.values()),
            max(t[1] + t[0]["w"] for t in placed.values()),
            max(t[2] + t[0]["h"] for t in placed.values()),
        )

    def finish(placed, portmap):
        pairs = []
        for values in portmap.values():
            if len(values) != 2:
                return
            a, b = values
            pairs.append(tuple(sorted((a[:3], b[:3]))))
        pattern = tuple(sorted(pairs))
        if pattern in m["failed"]:
            return
        lx, ly, rx, ry = bounds(placed)
        if rx - lx > W or ry - ly > H:
            return

        vx = sorted(pieces[i]["x"] - t[1] for i, t in placed.items())
        vy = sorted(pieces[i]["y"] - t[2] for i, t in placed.items())
        tx = max(-lx, min(W - rx, vx[len(vx) // 2]))
        ty = max(-ly, min(H - ry, vy[len(vy) // 2]))
        target, cost = {}, 0
        for i, (op, x, y) in placed.items():
            x, y = x + tx, y + ty
            p = pieces[i]
            moves = abs(x - p["x"]) + abs(y - p["y"])
            turns = (_hub_distance(p, op["v"], m["direction"])
                     if p["hub"] else (op["r"] - p["r"]) % 4)
            cost += moves + turns
            if (moves or turns) and i != m["active"]:
                cost += 1
            target[i] = op, x, y
        old = answers.get(pattern)
        if old is None or cost < old[0]:
            answers[pattern] = cost, target

    def add_piece(portmap, i, op, x, y):
        new, matches = portmap.copy(), 0
        for pid, (dx, dy) in enumerate(op["ports"]):
            key = x + dx, y + dy
            item = i, op["v"], pid, op["labels"][pid]
            old = new.get(key, ())
            if len(old) >= 2:
                return None
            if old:
                la, lb = old[0][3], item[3]
                if la >= 0 and lb >= 0 and la != lb:
                    return None
                matches += 1
            new[key] = tuple(old) + (item,)
        return new, matches

    def dfs(placed, remaining, portmap):
        nonlocal visits
        visits += 1
        if (visits > 80000 or time.monotonic() > deadline
                or len(answers) >= 250):
            return
        state = tuple(sorted((i, t[0]["v"], t[0]["r"], t[1], t[2])
                             for i, t in placed.items()))
        if state in seen:
            return
        seen.add(state)

        opened = [q for q, values in portmap.items() if len(values) == 1]
        if not remaining:
            if not opened:
                finish(placed, portmap)
            return
        maximum = sum(max(len(o["ports"]) for o in options[i]) for i in remaining)
        if len(opened) > maximum:
            return

        lx, ly, rx, ry = bounds(placed)
        candidates = None
        if opened:
            for q in opened:
                these, seen_local = [], set()
                for i in remaining:
                    for op in options[i]:
                        for dx, dy in op["ports"]:
                            x, y = q[0] - dx, q[1] - dy
                            key = i, op["v"], op["r"], x, y
                            if key in seen_local:
                                continue
                            seen_local.add(key)
                            if (max(rx, x + op["w"]) - min(lx, x) > W
                                    or max(ry, y + op["h"]) - min(ly, y) > H):
                                continue
                            added = add_piece(portmap, i, op, x, y)
                            if added is None:
                                continue
                            newmap, matches = added
                            p = pieces[i]
                            approximate = (
                                abs(pieces[anchor]["x"] + x - p["x"])
                                + abs(pieces[anchor]["y"] + y - p["y"]))
                            these.append((-matches, approximate, i, op, x, y, newmap))
                if not these:
                    return
                if candidates is None or len(these) < len(candidates):
                    candidates = these
                if len(candidates) <= 2:
                    break
        else:
            i = min(remaining)
            x0 = pieces[i]["x"] - pieces[anchor]["x"]
            y0 = pieces[i]["y"] - pieces[anchor]["y"]
            candidates = []
            for op in options[i]:
                for sx, sy in ((0, 0), (4, 0), (-4, 0), (0, 4), (0, -4)):
                    x, y = x0 + sx, y0 + sy
                    if (max(rx, x + op["w"]) - min(lx, x) > W
                            or max(ry, y + op["h"]) - min(ly, y) > H):
                        continue
                    added = add_piece(portmap, i, op, x, y)
                    if added is not None:
                        candidates.append((0, abs(sx) + abs(sy), i, op,
                                           x, y, added[0]))

        candidates.sort(key=lambda t: (t[0], t[1]))
        for _, _, i, op, x, y, newmap in candidates:
            placed[i] = op, x, y
            dfs(placed, remaining - {i}, newmap)
            del placed[i]
            if time.monotonic() > deadline:
                break

    for op in options[anchor]:
        added = add_piece({}, anchor, op, 0, 0)
        if added is not None:
            dfs({anchor: (op, 0, 0)}, set(range(n)) - {anchor}, added[0])
        if time.monotonic() > deadline:
            break
    if not answers:
        return None
    pattern, (_, target) = min(answers.items(), key=lambda item: item[1][0])
    return pattern, target


def _at_target(p, target):
    op, x, y = target
    return (p["x"] == x and p["y"] == y
            and p["v"] == op["v"] and p["r"] == op["r"])


def _all_at(m):
    return all(_at_target(p, m["target"][i])
               for i, p in enumerate(m["pieces"]))


def _move(m, i, action):
    p, op = m["pieces"][i], _option(m["pieces"][i])
    dx, dy = {
        "ACTION1": (0, -1), "ACTION2": (0, 1),
        "ACTION3": (-1, 0), "ACTION4": (1, 0),
    }[action]
    x, y = p["x"] + dx, p["y"] + dy
    if 0 <= x <= m["W"] - op["w"] and 0 <= y <= m["H"] - op["h"]:
        p["x"], p["y"] = x, y
    m["lastcheck"] = _all_at(m)
    return action, None, None


def _turn(m, i):
    p = m["pieces"][i]
    if p["hub"]:
        n, d = len(p["variants"]), m["direction"]
        nv = p["v"] + d
        if nv >= n or nv < 0:
            d = -d
            nv = p["v"] + d
        if 0 <= nv < n:
            p["v"] = nv
        m["direction"] = d
    else:
        p["r"] = (p["r"] + 1) % 4
    op = _option(p)
    p["x"] = max(0, min(p["x"], m["W"] - op["w"]))
    p["y"] = max(0, min(p["y"], m["H"] - op["h"]))
    m["lastcheck"] = _all_at(m)
    return "ACTION5", None, None


def act(obs, available, memory):
    m = memory
    obs = np.asarray(obs)
    if m.get("pieces") and int(obs[-1, -1]) != m["bg"]:
        m.clear()
    g = _decode(obs, m)

    if "pieces" not in m:
        if not _initialize(g, m):
            if "RESET" in available:
                return "RESET", None, None
            return "ACTION5", None, None

    pieces = m["pieces"]

    if "probe" in m:
        i, p = m["probe"], pieces[m["probe"]]
        old = p["variants"][p["v"]]
        captured = _capture_selected(g, m, p)
        if captured is not None:
            x, y, sh, zeros = captured
            p["x"], p["y"], p["zeros"] = x, y, zeros
            if not p["hub"]:
                is_rotation = (
                    set(_rot(old["cells"], old["w"], old["h"], 1)) == set(sh["cells"])
                    and set(_rot(old["ports"], old["w"], old["h"], 1)) == set(sh["ports"]))
                if is_rotation and not old["hint"]:
                    p.update(variants=[sh], r=0, v=0, tested=True)
                    del m["probe"]
                else:
                    p["hub"] = True
                    p["variants"].append(sh)
                    p["v"], p["r"] = 1, 0
            else:
                same = next((j for j, v in enumerate(p["variants"])
                             if _signature(v) == _signature(sh)), None)
                if same is not None:
                    previous = p["v"]
                    p["v"] = same
                    m["direction"] = -1 if same < previous else 1
                    p["tested"] = True
                    del m["probe"]
                else:
                    p["variants"].append(sh)
                    p["v"] = len(p["variants"]) - 1

            if "probe" in m:
                if (sh["final_hint"] or (m["grey"] and zeros == 0)
                        or len(p["variants"]) >= 12):
                    p["tested"] = True
                    del m["probe"]
        else:
            p["tested"] = True
            del m["probe"]
        m["active"] = i
        if "probe" in m:
            return "ACTION5", None, None
    else:
        m["active"] = _active(g, m)

    if m["phase"] == "learn":
        active = m["active"]
        if active is not None:
            p = pieces[active]
            if not p["learned"]:
                _capture_existing(g, p)
            if _needs_probe(p, m["grey"]):
                m["probe"] = active
                return "ACTION5", None, None
            p["tested"] = True

        unknown = [i for i, p in enumerate(pieces) if not p["learned"]]
        if unknown:
            if active is None:
                i = min(unknown, key=lambda j: pieces[j]["x"] ** 2 + pieces[j]["y"] ** 2)
            else:
                a = pieces[active]
                i = min(unknown, key=lambda j:
                        abs(pieces[j]["x"] - a["x"]) + abs(pieces[j]["y"] - a["y"]))
            return _click(m, i)

        untested = [i for i, p in enumerate(pieces) if _needs_probe(p, m["grey"])]
        if untested:
            return _click(m, untested[0])
        m["phase"] = "plan"

    if m["phase"] == "execute" and m.get("lastcheck", False):
        m["failed"].add(m["pattern"])
        m["lastcheck"] = False
        m["phase"] = "plan"

    if m["phase"] == "plan":
        result = _search(m)
        if result is None:
            if "RESET" in available and not m.get("reset_used", False):
                m.clear()
                m["reset_used"] = True
                return "RESET", None, None
            active = m["active"]
            if active is not None and not pieces[active]["hub"]:
                p = pieces[active]
                p["r"] = (p["r"] + 1) % 4
                op = _option(p)
                p["x"] = max(0, min(p["x"], m["W"] - op["w"]))
                p["y"] = max(0, min(p["y"], m["H"] - op["h"]))
                return "ACTION5", None, None
            return _click(m, 0)
        m["pattern"], m["target"] = result
        m["phase"], m["lastcheck"] = "execute", False

    pending = [i for i, p in enumerate(pieces)
               if not _at_target(p, m["target"][i])]
    if pending:
        active = m["active"]
        if active in pending:
            i = active
        else:
            def priority(i):
                op, x, y = m["target"][i]
                dest = {(x + dx, y + dy) for dx, dy in op["cells"]}
                interference = 0
                for j in pending:
                    if j != i:
                        q = pieces[j]
                        interference += sum(
                            (q["x"] + dx, q["y"] + dy) in dest
                            for dx, dy in _option(q)["cells"])
                p = pieces[i]
                return interference, abs(x - p["x"]) + abs(y - p["y"])
            return _click(m, min(pending, key=priority))

        p = pieces[i]
        op, x, y = m["target"][i]
        if p["v"] != op["v"] or p["r"] != op["r"]:
            return _turn(m, i)
        if p["x"] != x:
            return _move(m, i, "ACTION4" if p["x"] < x else "ACTION3")
        if p["y"] != y:
            return _move(m, i, "ACTION2" if p["y"] < y else "ACTION1")

    # Clicking does not check completion. Use a blocked movement, or briefly
    # move away and return on the next call.
    i = m["active"]
    if i is None:
        return _click(m, 0)
    p, op = pieces[i], _option(pieces[i])
    if p["x"] == 0:
        action = "ACTION3"
    elif p["y"] == 0:
        action = "ACTION1"
    elif p["x"] + op["w"] == m["W"]:
        action = "ACTION4"
    elif p["y"] + op["h"] == m["H"]:
        action = "ACTION2"
    else:
        action = "ACTION3"
    return _move(m, i, action)

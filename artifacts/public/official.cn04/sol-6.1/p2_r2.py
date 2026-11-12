import numpy as np
import math
import time
from collections import Counter, deque


# Visible templates help recover terminal colours that the renderer hides.
# Unrecognized pieces are handled geometrically instead.
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
    [".b8...", "8b....", ".bbd..", "8b....", ".bbbb8", "8b....",
     ".0...."],
    [".b8...", "8b....", ".bbd..", "8b....", ".bbbb8", "8b....",
     ".bbbd."],
    ["8.8.8", "e.e.e", "eeeee"],
    ["8aaaa", "....a", ".daaa"],
    [".8c", "..c", "dcc"],
]


def _shape(w, h, cells, pins):
    return {
        "w": int(w), "h": int(h),
        "cells": tuple(sorted(cells)),
        "pins": tuple(sorted(pins)),
    }


def _template(rows):
    cells, pins = [], []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                cells.append((x, y))
                if ch in ("8", "d"):
                    pins.append((x, y, 0 if ch == "8" else 1))
    return _shape(len(rows[0]), len(rows), cells, pins)


_TEMPLATES = tuple(_template(r) for r in _TEMPLATE_ROWS)


def _rotate(s, r=1):
    r %= 4
    out = s
    for _ in range(r):
        h = out["h"]
        out = _shape(
            h, out["w"],
            [(h - 1 - y, x) for x, y in out["cells"]],
            [(h - 1 - y, x, t) for x, y, t in out["pins"]],
        )
    return out


def _fingerprint(s):
    return (s["w"], s["h"], s["cells"],
            tuple((x, y) for x, y, _ in s["pins"]))


def _recognize(s):
    key = _fingerprint(s)
    for k, template in enumerate(_TEMPLATES):
        for r in range(4):
            t = _rotate(template, r)
            if _fingerprint(t) == key:
                return k, r, t
    return None


def _components(mask):
    h, w = mask.shape
    seen = set()
    result = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or (x, y) in seen:
                continue
            q = [(x, y)]
            seen.add((x, y))
            comp = []
            for px, py in q:
                comp.append((px, py))
                for nx, ny in ((px - 1, py), (px + 1, py),
                               (px, py - 1), (px, py + 1)):
                    if (0 <= nx < w and 0 <= ny < h
                            and mask[ny, nx] and (nx, ny) not in seen):
                        seen.add((nx, ny))
                        q.append((nx, ny))
            result.append(comp)
    return result


def _board(obs, memory):
    scale = memory["scale"]
    off = memory["off"]
    n = memory["size"]
    center = off + scale // 2
    return np.asarray(obs)[center:center + n * scale:scale,
                           center:center + n * scale:scale]


def _geometry(obs):
    a = np.asarray(obs)
    bg = Counter(map(int, a[1:].ravel())).most_common(1)[0][0]
    lengths = []
    for row in a[2:]:
        changes = np.flatnonzero(row[1:] != row[:-1]) + 1
        if len(changes) >= 2:
            lengths.extend(int(v) for v in np.diff(changes) if v > 0)
    g = 0
    for length in lengths:
        g = math.gcd(g, length)
    scale = g if g in (2, 3, 4) else 3
    n = 64 // scale
    off = (64 - n * scale) // 2
    return bg, scale, n, off


def _current_shape(obj):
    if obj.get("group"):
        return obj["options"][obj["v"]]
    return _rotate(obj["options"][0], obj["r"])


def _capture(board, obj, memory, fixed=False):
    bg = memory["bg"]
    if fixed:
        s = _current_shape(obj)
        x, y = obj["x"], obj["y"]
        pins = []
        for px, py in s["cells"]:
            xx, yy = x + px, y + py
            if (0 <= yy < board.shape[0] and 0 <= xx < board.shape[1]
                    and int(board[yy, xx]) in (8, 3)):
                pins.append((px, py, -1))
        return _shape(s["w"], s["h"], s["cells"], pins), x, y

    if memory["grey"]:
        mask = (board != bg) & (board != 4)
    else:
        mask = np.isin(board, (0, 8, 3))
        mask &= board != bg

    comps = _components(mask)
    if not comps:
        return _capture(board, obj, memory, True)

    def score(c):
        body = sum(int(board[y, x]) not in (8, 3) for x, y in c)
        cx = sum(x for x, _ in c) / len(c)
        cy = sum(y for _, y in c) / len(c)
        s = _current_shape(obj)
        d = abs(cx - obj["x"] - s["w"] / 2)
        d += abs(cy - obj["y"] - s["h"] / 2)
        return body * 20 + len(c) - d * 2

    comp = max(comps, key=score)
    x = min(px for px, _ in comp)
    y = min(py for _, py in comp)
    w = max(px for px, _ in comp) - x + 1
    h = max(py for _, py in comp) - y + 1
    cells = [(px - x, py - y) for px, py in comp]
    pins = [(px - x, py - y, -1) for px, py in comp
            if int(board[py, px]) in (8, 3)]
    return _shape(w, h, cells, pins), x, y


def _initialize(obs, memory):
    bg, scale, n, off = _geometry(obs)
    memory.update(bg=bg, scale=scale, size=n, off=off)
    board = _board(obs, memory)
    comps = _components(board != bg)
    comps = [c for c in comps if len(c) >= 2]
    comps.sort(key=lambda c: (min(y for _, y in c), min(x for x, _ in c)))

    colors = Counter(int(board[y, x]) for c in comps for x, y in c)
    other_body = sum(v for k, v in colors.items()
                     if k not in (0, 3, 4, 8))
    memory["grey"] = colors[4] > 0 and colors[4] >= other_body

    objects = []
    for c in comps:
        x, y = min(px for px, _ in c), min(py for _, py in c)
        w = max(px for px, _ in c) - x + 1
        h = max(py for _, py in c) - y + 1
        s = _shape(w, h, [(px - x, py - y) for px, py in c], [])
        objects.append({
            "x": x, "y": y, "r": 0, "v": 0,
            "group": False, "options": [s], "learned": False,
        })

    selected = None
    best = 0
    for i, c in enumerate(comps):
        if memory["grey"]:
            value = sum(int(board[y, x]) not in (4, 8, 3)
                        for x, y in c)
        else:
            value = sum(int(board[y, x]) == 0 for x, y in c)
        if value > best:
            best, selected = value, i

    memory.update(objects=objects, selected=selected, phase="learn",
                  failed=set(), plan=None, global_dir=1, last=None)
    return board


def _click(i, memory):
    objects = memory["objects"]
    obj = objects[i]
    s = _current_shape(obj)
    occupied = Counter()
    for j, other in enumerate(objects):
        if i == j:
            continue
        for x, y in _current_shape(other)["cells"]:
            occupied[(other["x"] + x, other["y"] + y)] += 1
    pinset = {(x, y) for x, y, _ in s["pins"]}
    choices = []
    for x, y in s["cells"]:
        gx, gy = obj["x"] + x, obj["y"] + y
        rank = (occupied[(gx, gy)], (x, y) in pinset, y, x)
        choices.append((rank, gx, gy))
    if not choices:
        return None
    _, x, y = min(choices)
    memory["selected"] = i
    memory["last"] = "ACTION6"
    scale, off = memory["scale"], memory["off"]
    return ("ACTION6", off + y * scale + scale // 2,
            off + x * scale + scale // 2)


def _emit(name, memory):
    memory["last"] = name
    return name, None, None


def _start_learning(board, memory):
    objects = memory["objects"]
    i = memory["selected"]
    if i is None or objects[i]["learned"]:
        todo = [j for j, o in enumerate(objects) if not o["learned"]]
        if not todo:
            memory["phase"] = "execute"
            return None
        return _click(todo[0], memory)

    obj = objects[i]
    s, x, y = _capture(board, obj, memory, True)
    match = _recognize(s)
    if match is not None:
        k, r, typed = match
        obj["x"], obj["y"] = x, y
        if 5 <= k <= 9:
            obj["group"] = True
            obj["options"] = [_rotate(_TEMPLATES[j], r)
                              for j in range(5, 10)]
            obj["v"] = k - 5
            obj["r"] = 0
        else:
            obj["options"] = [typed]
            obj["r"] = 0
        obj["learned"] = True
        return None

    obj["options"] = [s]
    obj["x"], obj["y"] = x, y
    obj["r"] = 0
    memory["phase"] = "verify"
    memory["verify_first"] = s
    memory["verify_xy"] = (x, y)
    return _emit("ACTION5", memory)


def _verify(board, memory):
    obj = memory["objects"][memory["selected"]]
    first = memory["verify_first"]
    actual, x, y = _capture(board, obj, memory)
    expected = _rotate(first)
    if _fingerprint(actual) == _fingerprint(expected):
        obj["r"] = 1
        obj["x"], obj["y"] = x, y
        obj["learned"] = True
        memory["phase"] = "learn"
        return None

    obj["group"] = True
    obj["r"] = 0
    obj["options"] = [first]
    if _fingerprint(actual) != _fingerprint(first):
        obj["options"].append(actual)
        obj["v"] = 1
        seq = [0, 1]
    else:
        obj["v"] = 0
        seq = [0, 0]
    obj["x"], obj["y"] = x, y
    memory["explore_seq"] = seq
    memory["explore_pairs"] = {(seq[0], seq[1]): 0}
    memory["phase"] = "explore"
    return _emit("ACTION5", memory)


def _explore(board, memory):
    obj = memory["objects"][memory["selected"]]
    actual, x, y = _capture(board, obj, memory)
    key = _fingerprint(actual)
    v = next((j for j, s in enumerate(obj["options"])
              if _fingerprint(s) == key), None)
    if v is None:
        match = _recognize(actual)
        if match is not None:
            actual = match[2]
        v = len(obj["options"])
        obj["options"].append(actual)
    obj["v"] = v
    obj["x"], obj["y"] = x, y

    seq = memory["explore_seq"]
    pair = (seq[-1], v)
    seq.append(v)
    pairs = memory["explore_pairs"]
    if pair in pairs or len(seq) >= 24:
        if pair in pairs:
            start = pairs[pair]
            period = len(seq) - 2 - start
            if period > 0:
                obj["cycle"] = seq[start:start + period]
                obj["cycle_pos"] = (len(seq) - 1 - start) % period
        obj["learned"] = True
        memory["phase"] = "learn"
        return None
    pairs[pair] = len(seq) - 2
    return _emit("ACTION5", memory)


def _compatible(a, b):
    return a < 0 or b < 0 or a == b


def _signature(targets):
    minx = min(t[2] for t in targets)
    miny = min(t[3] for t in targets)
    return tuple((v, r, x - minx, y - miny)
                 for v, r, x, y in targets)


def _solve(memory):
    objects = memory["objects"]
    n = len(objects)
    size = memory["size"]
    if not n or n > 9:
        return None

    configs = []
    for obj in objects:
        choices = []
        for v, base in enumerate(obj["options"]):
            rotations = (0,) if obj["group"] else range(4)
            for r in rotations:
                s = _rotate(base, r)
                if not s["pins"] or s["w"] > size or s["h"] > size:
                    continue
                choices.append((v, r, s))
        configs.append(choices)
    if any(not c for c in configs):
        return None

    anchor = max(range(n), key=lambda i:
                 max(len(s["pins"]) for _, _, s in configs[i])
                 / (1 + 0.15 * len(configs[i])))
    failed = memory["failed"]
    deadline = time.perf_counter() + 5.7
    best = [float("inf"), None]
    nodes = [0]
    found = [0]

    def goal(placed, bounds):
        minx, miny, maxx, maxy = bounds
        xs = sorted(objects[i]["x"] - p[2] for i, p in placed.items())
        ys = sorted(objects[i]["y"] - p[3] for i, p in placed.items())
        sx = max(-minx, min(size - maxx, xs[len(xs) // 2]))
        sy = max(-miny, min(size - maxy, ys[len(ys) // 2]))
        targets = [None] * n
        cost = 0
        for i, (v, r, x, y, s) in placed.items():
            tx, ty = x + sx, y + sy
            targets[i] = (v, r, tx, ty)
            obj = objects[i]
            cost += abs(tx - obj["x"]) + abs(ty - obj["y"])
            if obj["group"]:
                cost += abs(v - obj["v"])
            else:
                cost += (r - obj["r"]) % 4
            if (v, r, tx, ty) != (obj["v"], obj["r"],
                                  obj["x"], obj["y"]):
                cost += int(memory["selected"] != i)
        if _signature(targets) in failed:
            return
        found[0] += 1
        if cost < best[0]:
            best[:] = [cost, targets]

    def search(placed, openpins, closed, occupied, bounds, strict):
        nodes[0] += 1
        if nodes[0] > 70000 or time.perf_counter() > deadline:
            return
        if found[0] >= 300:
            return
        if len(placed) == n:
            if not openpins:
                goal(placed, bounds)
            return
        if not openpins:
            return
        remaining = [i for i in range(n) if i not in placed]

        # A dangling terminal must be covered by a remaining piece.
        def possibilities(item):
            _, t = item
            return sum(sum(_compatible(t, pt) for _, _, pt in s["pins"])
                       for i in remaining for _, _, s in configs[i])

        point, typ = min(openpins.items(), key=possibilities)
        px, py = point
        candidates = []
        aminx, aminy, amaxx, amaxy = bounds
        anchor_x = objects[anchor]["x"]
        anchor_y = objects[anchor]["y"]

        for i in remaining:
            obj = objects[i]
            for v, r, s in configs[i]:
                pinlocal = {(x, y): t for x, y, t in s["pins"]}
                for cx, cy, ct in s["pins"]:
                    if not _compatible(typ, ct):
                        continue
                    x, y = px - cx, py - cy
                    b = (min(aminx, x), min(aminy, y),
                         max(amaxx, x + s["w"]),
                         max(amaxy, y + s["h"]))
                    if b[2] - b[0] > size or b[3] - b[1] > size:
                        continue
                    pins = {(x + qx, y + qy): qt
                            for (qx, qy), qt in pinlocal.items()}
                    if any(p in closed for p in pins):
                        continue
                    if any(p in openpins and
                           not _compatible(t, openpins[p])
                           for p, t in pins.items()):
                        continue
                    cells = {(x + qx, y + qy) for qx, qy in s["cells"]}
                    if strict:
                        overlap = cells & occupied
                        if any(p not in pins or p not in openpins
                               for p in overlap):
                            continue
                    approximate = abs(x + anchor_x - obj["x"])
                    approximate += abs(y + anchor_y - obj["y"])
                    approximate += ((r - obj["r"]) % 4
                                    if not obj["group"]
                                    else abs(v - obj["v"]))
                    matches = sum(p in openpins for p in pins)
                    approximate -= 3 * matches
                    candidates.append((approximate, i, v, r, x, y,
                                       s, pins, cells, b))

        candidates.sort(key=lambda c: c[0])
        seen = set()
        for _, i, v, r, x, y, s, pins, cells, b in candidates:
            key = (i, v, r, x, y)
            if key in seen:
                continue
            seen.add(key)
            newopen = dict(openpins)
            newclosed = set(closed)
            for p, t in pins.items():
                if p in newopen:
                    del newopen[p]
                    newclosed.add(p)
                else:
                    newopen[p] = t
            # Every open terminal still requires another terminal.
            remaining_pins = sum(
                max(len(z["pins"]) for _, _, z in configs[j])
                for j in remaining if j != i
            )
            if len(newopen) > remaining_pins:
                continue
            placed[i] = (v, r, x, y, s)
            search(placed, newopen, newclosed,
                   occupied | cells if strict else set(), b, strict)
            del placed[i]
            if nodes[0] > 70000 or time.perf_counter() > deadline:
                return

    anchor_choices = list(configs[anchor])
    anchor_choices.sort(key=lambda c:
                        abs(c[0] - objects[anchor]["v"]) +
                        ((c[1] - objects[anchor]["r"]) % 4))
    for strict in (True, False):
        nodes[0] = 0
        for v, r, s in anchor_choices:
            placed = {anchor: (v, r, 0, 0, s)}
            pins = {(x, y): t for x, y, t in s["pins"]}
            cells = set(s["cells"]) if strict else set()
            search(placed, pins, set(), cells,
                   (0, 0, s["w"], s["h"]), strict)
            if time.perf_counter() > deadline:
                break
        if best[1] is not None or time.perf_counter() > deadline:
            break
    return best[1]


def _move(obj, action, memory):
    s = _current_shape(obj)
    dx, dy = {
        "ACTION1": (0, -1), "ACTION2": (0, 1),
        "ACTION3": (-1, 0), "ACTION4": (1, 0),
    }[action]
    x, y = obj["x"] + dx, obj["y"] + dy
    if (0 <= x <= memory["size"] - s["w"]
            and 0 <= y <= memory["size"] - s["h"]):
        obj["x"], obj["y"] = x, y
    memory["had_check"] = True
    return _emit(action, memory)


def _group_step(obj, memory):
    old = obj["v"]
    expected = old
    if obj.get("cycle"):
        pos = (obj.get("cycle_pos", 0) + 1) % len(obj["cycle"])
        obj["cycle_pos"] = pos
        expected = obj["cycle"][pos]
    else:
        direction = memory.get("global_dir", 1)
        expected = old + direction
        if expected >= len(obj["options"]):
            expected = max(0, old - 1)
        if expected < 0:
            expected = min(len(obj["options"]) - 1, old + 1)
    memory["pending_group"] = (memory["selected"], old, expected)
    memory["had_check"] = True
    return _emit("ACTION5", memory)


def _read_group(board, memory):
    pending = memory.pop("pending_group", None)
    if pending is None:
        return
    i, old, expected = pending
    obj = memory["objects"][i]
    actual, x, y = _capture(board, obj, memory)
    key = _fingerprint(actual)
    v = next((j for j, s in enumerate(obj["options"])
              if _fingerprint(s) == key), None)
    if v is None:
        v = expected
        s = obj["options"][v]
        x = min(obj["x"], memory["size"] - s["w"])
        y = min(obj["y"], memory["size"] - s["h"])
    elif not obj.get("cycle") and v != old:
        memory["global_dir"] = 1 if v > old else -1
    obj["v"], obj["x"], obj["y"] = v, x, y
    if obj.get("cycle"):
        p = obj.get("cycle_pos", 0)
        if obj["cycle"][p] != v:
            options = [j for j, z in enumerate(obj["cycle"]) if z == v]
            if options:
                obj["cycle_pos"] = options[0]


def _execute(board, memory):
    objects = memory["objects"]
    _read_group(board, memory)

    restore = memory.pop("probe_restore", None)
    if restore is not None:
        return _move(objects[memory["selected"]], restore, memory)

    for _ in range(3):
        if memory.get("plan") is None:
            plan = _solve(memory)
            if plan is None:
                return None
            memory["plan"] = plan
            memory["had_check"] = False
        plan = memory["plan"]

        unfinished = [
            i for i, o in enumerate(objects)
            if (o["v"], o["r"], o["x"], o["y"]) != tuple(plan[i])
        ]
        if not unfinished:
            if memory.get("had_check"):
                memory["failed"].add(_signature(plan))
                memory["plan"] = None
                continue

            i = memory["selected"]
            if i is None:
                return _click(0, memory)
            obj = objects[i]
            s = _current_shape(obj)
            if obj["y"] == 0:
                return _move(obj, "ACTION1", memory)
            if obj["x"] == 0:
                return _move(obj, "ACTION3", memory)
            if obj["y"] + s["h"] == memory["size"]:
                return _move(obj, "ACTION2", memory)
            if obj["x"] + s["w"] == memory["size"]:
                return _move(obj, "ACTION4", memory)
            memory["probe_restore"] = "ACTION2"
            return _move(obj, "ACTION1", memory)

        selected = memory["selected"]
        if selected in unfinished:
            i = selected
        else:
            i = min(unfinished, key=lambda j:
                    abs(objects[j]["x"] - plan[j][2]) +
                    abs(objects[j]["y"] - plan[j][3]))
            return _click(i, memory)

        obj = objects[i]
        v, r, tx, ty = plan[i]
        if obj["group"] and obj["v"] != v:
            return _group_step(obj, memory)
        if not obj["group"] and obj["r"] != r:
            obj["r"] = (obj["r"] + 1) % 4
            s = _current_shape(obj)
            obj["x"] = min(obj["x"], memory["size"] - s["w"])
            obj["y"] = min(obj["y"], memory["size"] - s["h"])
            memory["had_check"] = True
            return _emit("ACTION5", memory)
        if obj["x"] < tx:
            return _move(obj, "ACTION4", memory)
        if obj["x"] > tx:
            return _move(obj, "ACTION3", memory)
        if obj["y"] < ty:
            return _move(obj, "ACTION2", memory)
        if obj["y"] > ty:
            return _move(obj, "ACTION1", memory)
    return None


def act(obs, available, memory):
    available = set(available)

    def fallback_reset():
        memory.clear()
        if "RESET" in available:
            return "RESET", None, None
        for name in ("ACTION5", "ACTION1", "ACTION2",
                     "ACTION3", "ACTION4"):
            if name in available:
                return name, None, None
        return "ACTION1", None, None

    try:
        if not memory or "objects" not in memory:
            board = _initialize(obs, memory)
        else:
            board = _board(obs, memory)

        if not memory["objects"]:
            return fallback_reset()

        result = None
        for _ in range(12):
            phase = memory["phase"]
            if phase == "learn":
                result = _start_learning(board, memory)
            elif phase == "verify":
                result = _verify(board, memory)
            elif phase == "explore":
                result = _explore(board, memory)
            else:
                result = _execute(board, memory)
                if result is None:
                    return fallback_reset()
            if result is not None:
                if result[0] in available:
                    return result
                return fallback_reset()

        return fallback_reset()
    except (ValueError, IndexError, KeyError, TypeError, ZeroDivisionError):
        return fallback_reset()

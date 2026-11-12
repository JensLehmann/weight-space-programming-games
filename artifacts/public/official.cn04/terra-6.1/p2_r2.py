import time
from collections import deque

# "." is transparent; "8" and "d" are the two original connector types.
_TEMPLATES = [
    ("a", ["ccc...", "c.ccc8", "c.....", "c.cccd", "ccc..."]),
    ("b", [".eee", ".e.e", "8e.e", ".e.e", "de.e", ".e.e", ".eee"]),
    ("c", ["db", ".b", ".b", ".b", ".b", ".d"]),
    ("d", ["8.....d", "e.....e", "eee.eee", "..e.e..", "..eee.."]),
    ("e", ["ffffff.", "f....fd", "f.ffff.", "f.f....",
           "f.f....", "fff....", ".8....."]),
    ("m0", [".b8.", "8b..", ".bbd", ".0.."]),
    ("m1", [".b8.", "8b..", ".bbd", "8b..", ".0.."]),
    ("m2", [".b8...", "8b....", ".bbd..", "8b....", ".bbbb8", ".0...."]),
    ("m3", [".b8...", "8b....", ".bbd..", "8b....",
            ".bbbb8", "8b....", ".0...."]),
    ("m4", [".b8...", "8b....", ".bbd..", "8b....",
            ".bbbb8", "8b....", ".bbbd."]),
    ("f", ["8.8.8", "e.e.e", "eeeee"]),
    ("g", ["8aaaa", "....a", ".daaa"]),
    ("h", [".8c", "..c", "dcc"]),
]


def _state_from_strings(rows):
    cells = []
    ports = []
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c != ".":
                cells.append((x, y))
                if c == "8":
                    ports.append((x, y, 8))
                elif c == "d":
                    ports.append((x, y, 13))
    return {
        "w": len(rows[0]), "h": len(rows),
        "cells": cells, "ports": ports,
    }


_KNOWN = [(name, _state_from_strings(rows)) for name, rows in _TEMPLATES]
_MORPH = [dict(s) for name, s in _KNOWN if name.startswith("m")]


def _rotate(s):
    h = s["h"]
    return {
        "w": h, "h": s["w"],
        "cells": [(h - 1 - y, x) for x, y in s["cells"]],
        "ports": [(h - 1 - y, x, t) for x, y, t in s["ports"]],
    }


def _shape_key(s):
    return (
        s["w"], s["h"],
        frozenset(s["cells"]),
        frozenset((x, y) for x, y, _ in s["ports"]),
    )


def _identify(s):
    key = _shape_key(s)
    for name, original in _KNOWN:
        q = original
        for r in range(4):
            if _shape_key(q) == key:
                return name, r, q
            q = _rotate(q)
    return None, None, None


def _grid(obs):
    # The canonical camera uses a 20 by 20 grid, scaled by 3 and
    # centered with a two-pixel border.
    return [[int(obs[3 + 3 * y, 3 + 3 * x]) for x in range(20)]
            for y in range(20)]


def _components(g, bg):
    seen = set()
    result = []
    for y in range(20):
        for x in range(20):
            if g[y][x] == bg or (x, y) in seen:
                continue
            seen.add((x, y))
            todo = [(x, y)]
            part = []
            while todo:
                a, b = todo.pop()
                part.append((a, b))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c, d = a + dx, b + dy
                    if (0 <= c < 20 and 0 <= d < 20 and
                            (c, d) not in seen and g[d][c] != bg):
                        seen.add((c, d))
                        todo.append((c, d))
            result.append(part)
    return result


def _selected_component(g, parts):
    # With no grey mask the selected body is black. With a grey mask,
    # all other bodies are grey and the selected body keeps its colour.
    for i, part in enumerate(parts):
        colors = {g[y][x] for x, y in part}
        if 0 in colors and colors <= {0, 3, 8}:
            return i, False
    best = None
    for i, part in enumerate(parts):
        colors = {g[y][x] for x, y in part}
        score = sum(g[y][x] not in (4, 3) for x, y in part)
        if score and (best is None or score > best[0]):
            best = (score, i)
    return (best[1], True) if best else (0, True)


def _read_piece(g, part, grey):
    x0 = min(x for x, y in part)
    y0 = min(y for x, y in part)
    x1 = max(x for x, y in part)
    y1 = max(y for x, y in part)
    cells = [(x - x0, y - y0) for x, y in part]
    ports = [(x - x0, y - y0, -1) for x, y in part
             if g[y][x] in (8, 3)]
    ports.sort()
    s = {"w": x1 - x0 + 1, "h": y1 - y0 + 1,
         "cells": cells, "ports": ports}
    name, rotation, known = _identify(s)
    if known is not None:
        # Keep connector IDs in a stable, deterministic order.
        s = {
            "w": known["w"], "h": known["h"],
            "cells": list(known["cells"]),
            "ports": sorted(known["ports"]),
        }
    colored_body = any(g[y][x] not in (0, 3, 4, 8) for x, y in part)
    sentinel = grey and colored_body and any(g[y][x] == 0 for x, y in part)
    p = {
        "x": x0, "y": y0, "base": s, "ori": 0,
        "morph": False, "sentinel": sentinel, "name": name,
    }
    if name is not None and name.startswith("m") and rotation == 0:
        p["morph"] = True
        p["states"] = _MORPH
        p["stage"] = int(name[1:])
        p["direction"] = 1
        p["sentinel"] = False
    return p


def _current_state(p):
    if p["morph"]:
        return p["states"][p["stage"]]
    s = p["base"]
    for _ in range(p["ori"]):
        s = _rotate(s)
    return s


def _advance_stage(stage, direction, n):
    if n <= 1:
        return stage, direction
    nxt = stage + direction
    if nxt >= n or nxt < 0:
        direction = -direction
        nxt = stage + direction
    return nxt, direction


def _options(p):
    if p["morph"]:
        result = []
        st, direction = p["stage"], p["direction"]
        seen = set()
        for steps in range(max(1, 2 * len(p["states"]) + 1)):
            if st not in seen:
                result.append((st, steps, p["states"][st]))
                seen.add(st)
            st, direction = _advance_stage(st, direction, len(p["states"]))
        return result
    result = []
    s = p["base"]
    for ori in range(4):
        result.append((ori, (ori - p["ori"]) % 4, s))
        s = _rotate(s)
    return result


def _port_id(i, p, option, j):
    # Different morph stages can contain different numbers of connectors.
    return (i, option if p["morph"] else 0, j)


def _failed_implied(pairs, failures):
    if not failures:
        return False
    parent = {}

    def find(a):
        if a not in parent:
            parent[a] = a
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for failed in failures:
        if failed and all(find(a) == find(b) for a, b in failed):
            return True
    return False


def _solve(pieces, selected, failures):
    n = len(pieces)
    if not n:
        return None

    opts = [_options(p) for p in pieces]
    deadline = time.monotonic() + 2.5
    nodes = [0]
    best = [None, float("inf")]
    anchor = max(range(n), key=lambda i: len(_current_state(pieces[i])["ports"]))

    def bounding_ok(placed):
        left = min(x for x, y, opt, s in placed.values())
        top = min(y for x, y, opt, s in placed.values())
        right = max(x + s["w"] for x, y, opt, s in placed.values())
        bottom = max(y + s["h"] for x, y, opt, s in placed.values())
        return right - left <= 20 and bottom - top <= 20

    def evaluate(placed, pairs):
        if _failed_implied(pairs, failures):
            return
        minx = min(x for x, y, op, s in placed.values())
        miny = min(y for x, y, op, s in placed.values())
        maxx = max(x + s["w"] for x, y, op, s in placed.values())
        maxy = max(y + s["h"] for x, y, op, s in placed.values())
        lo_x, hi_x = -minx, 20 - maxx
        lo_y, hi_y = -miny, 20 - maxy
        if lo_x > hi_x or lo_y > hi_y:
            return
        xs = sorted(pieces[i]["x"] - placed[i][0] for i in placed)
        ys = sorted(pieces[i]["y"] - placed[i][1] for i in placed)
        tx = max(lo_x, min(hi_x, xs[len(xs) // 2]))
        ty = max(lo_y, min(hi_y, ys[len(ys) // 2]))
        cost = 0
        targets = [None] * n
        for i, (x, y, option, s) in placed.items():
            p = pieces[i]
            turns = next(k for o, k, state in opts[i] if o == option)
            distance = abs(x + tx - p["x"]) + abs(y + ty - p["y"])
            cost += distance + turns
            if distance or turns:
                cost += 1 if i != selected else 0
            targets[i] = (x + tx, y + ty, option)
        if cost < best[1]:
            best[0] = {"targets": targets, "pairs": list(pairs)}
            best[1] = cost

    def rec(placed, occupied, pairs):
        nodes[0] += 1
        if nodes[0] > 100000 or (
                nodes[0] % 256 == 0 and time.monotonic() > deadline):
            return

        if len(placed) == n:
            if all(len(v) == 2 for v in occupied.values()):
                evaluate(placed, pairs)
            return

        free = [(pos, v[0]) for pos, v in occupied.items() if len(v) == 1]
        unused = [i for i in range(n) if i not in placed]
        if free:
            remaining_max = sum(max(len(s["ports"]) for _, _, s in opts[i])
                                for i in unused)
            if len(free) > remaining_max:
                return
            # Prefer a known-type connector when available.
            free.sort(key=lambda z: z[1][1] == -1)
            (fx, fy), (fid, ft) = free[0]
            candidates = []
            for i in unused:
                p = pieces[i]
                for option, turns, s in opts[i]:
                    for j, (px, py, pt) in enumerate(s["ports"]):
                        if ft != -1 and pt != -1 and ft != pt:
                            continue
                        x, y = fx - px, fy - py
                        estimate = abs(x - p["x"]) + abs(y - p["y"]) + turns
                        candidates.append((estimate, i, option, s, x, y))
            candidates.sort(key=lambda z: z[0])
        else:
            # Begin a disconnected cluster near its current relative position.
            i = unused[0]
            p = pieces[i]
            ax, ay, _, _ = placed[anchor]
            x = ax + p["x"] - pieces[anchor]["x"]
            y = ay + p["y"] - pieces[anchor]["y"]
            candidates = [(turns, i, option, s, x, y)
                          for option, turns, s in opts[i]]

        for _, i, option, s, x, y in candidates:
            if time.monotonic() > deadline or nodes[0] > 100000:
                break
            new_entries = []
            new_pairs = []
            valid = True
            for j, (px, py, pt) in enumerate(s["ports"]):
                pos = (x + px, y + py)
                ident = _port_id(i, pieces[i], option, j)
                old = occupied.get(pos, [])
                if len(old) >= 2:
                    valid = False
                    break
                if old:
                    old_id, old_type = old[0]
                    if pt != -1 and old_type != -1 and pt != old_type:
                        valid = False
                        break
                    new_pairs.append((old_id, ident))
                new_entries.append((pos, (ident, pt)))
            if not valid:
                continue
            placed[i] = (x, y, option, s)
            if not bounding_ok(placed):
                del placed[i]
                continue
            for pos, entry in new_entries:
                occupied.setdefault(pos, []).append(entry)
            rec(placed, occupied, pairs + new_pairs)
            for pos, entry in new_entries:
                occupied[pos].pop()
                if not occupied[pos]:
                    del occupied[pos]
            del placed[i]

    for option, turns, state in opts[anchor]:
        occupied = {}
        for j, (x, y, t) in enumerate(state["ports"]):
            occupied[(x, y)] = [(_port_id(anchor, pieces[anchor], option, j), t)]
        rec({anchor: (0, 0, option, state)}, occupied, [])
        if time.monotonic() > deadline:
            break
    return best[0]


def _click_for_piece(pieces, i):
    p = pieces[i]
    s = _current_state(p)
    other = set()
    for j, q in enumerate(pieces):
        if j == i:
            continue
        t = _current_state(q)
        other.update((q["x"] + x, q["y"] + y) for x, y in t["cells"])
    choices = [(p["x"] + x, p["y"] + y) for x, y in s["cells"]]
    choices.sort(key=lambda xy: (xy in other, xy[1], xy[0]))
    if not choices:
        return 3, 3
    x, y = choices[0]
    return 3 + 3 * y, 3 + 3 * x


def _rotate_piece(p):
    if p["morph"]:
        p["stage"], p["direction"] = _advance_stage(
            p["stage"], p["direction"], len(p["states"]))
    else:
        p["ori"] = (p["ori"] + 1) % 4
    s = _current_state(p)
    p["x"] = max(0, min(p["x"], 20 - s["w"]))
    p["y"] = max(0, min(p["y"], 20 - s["h"]))


def act(obs, available, memory):
    if not memory:
        memory["phase"] = "reset"
        # A non-reset action first ensures RESET is a level reset rather
        # than a full-game reset when the episode begins at action count 0.
        return ("ACTION1", None, None)

    if memory["phase"] == "reset":
        memory["phase"] = "initialize"
        if "RESET" in available:
            return ("RESET", None, None)

    g = _grid(obs)

    if memory["phase"] == "initialize":
        counts = {}
        for row in g:
            for c in row:
                counts[c] = counts.get(c, 0) + 1
        bg = max(counts, key=counts.get)
        parts = _components(g, bg)
        if not parts:
            return ("ACTION1", None, None)
        selected, grey = _selected_component(g, parts)
        memory["bg"] = bg
        memory["parts"] = parts
        memory["grey"] = grey
        memory["pieces"] = [None] * len(parts)
        memory["pieces"][selected] = _read_piece(g, parts[selected], grey)
        memory["selected"] = selected
        memory["pending"] = None
        memory["phase"] = "scan"

    if memory["phase"] == "scan":
        pending = memory.get("pending")
        if pending is not None:
            memory["pieces"][pending] = _read_piece(
                g, memory["parts"][pending], memory["grey"])
            memory["selected"] = pending
            memory["pending"] = None
        for i, p in enumerate(memory["pieces"]):
            if p is None:
                x, y = memory["parts"][i][0]
                memory["pending"] = i
                return ("ACTION6", 3 + 3 * y, 3 + 3 * x)
        memory["probe_list"] = [
            i for i, p in enumerate(memory["pieces"])
            if p["sentinel"] and not p["morph"]
        ]
        memory["phase"] = "probe_select"

    if memory["phase"] == "probe_select":
        if memory["probe_list"]:
            i = memory["probe_list"].pop(0)
            memory["probe_i"] = i
            if memory["selected"] != i:
                row, col = _click_for_piece(memory["pieces"], i)
                memory["selected"] = i
                memory["phase"] = "probe_start"
                return ("ACTION6", row, col)
            memory["phase"] = "probe_start"
        else:
            memory["phase"] = "plan"
            memory["failures"] = []

    if memory["phase"] == "probe_start":
        i = memory["probe_i"]
        p = memory["pieces"][i]
        memory["probe_states"] = [p["base"]]
        memory["probe_previous"] = p["base"]
        memory["phase"] = "probe_result"
        return ("ACTION5", None, None)

    if memory["phase"] == "probe_result":
        i = memory["probe_i"]
        parts = _components(g, memory["bg"])
        if not parts:
            memory["phase"] = "probe_select"
            return ("ACTION1", None, None)
        si, _ = _selected_component(g, parts)
        observed = _read_piece(g, parts[si], memory["grey"])
        p = memory["pieces"][i]
        p["x"], p["y"] = observed["x"], observed["y"]
        s = observed["base"]
        states = memory["probe_states"]
        if len(states) == 1 and _shape_key(s) == _shape_key(_rotate(states[0])):
            # Ordinary rotation, not a shape-changing piece.
            p["base"] = s
            p["ori"] = 0
            p["sentinel"] = False
            memory["phase"] = "probe_select"
        else:
            match = next((j for j, old in enumerate(states)
                          if _shape_key(old) == _shape_key(s)), None)
            if match is not None or len(states) >= 10:
                if match is None:
                    states.append(s)
                    match = len(states) - 1
                p["morph"] = True
                p["states"] = states
                p["stage"] = match
                p["direction"] = -1 if match < len(states) - 1 else 1
                p["sentinel"] = False
                memory["phase"] = "probe_select"
            else:
                states.append(s)
                return ("ACTION5", None, None)

    if memory["phase"] == "plan":
        plan = _solve(memory["pieces"], memory["selected"], memory["failures"])
        if plan is None:
            # If no placement was found, try exposing a different shape.
            p = memory["pieces"][memory["selected"]]
            _rotate_piece(p)
            return ("ACTION5", None, None)
        memory["plan"] = plan
        memory["last_check"] = False
        memory["phase"] = "execute"

    if memory["phase"] == "execute":
        pieces = memory["pieces"]
        targets = memory["plan"]["targets"]
        unfinished = []
        for i, p in enumerate(pieces):
            x, y, option = targets[i]
            current_option = p["stage"] if p["morph"] else p["ori"]
            if (p["x"], p["y"], current_option) != (x, y, option):
                unfinished.append(i)

        if not unfinished:
            if memory["last_check"]:
                memory["failures"].append(memory["plan"]["pairs"])
                memory["phase"] = "plan"
                return act(obs, available, memory)
            # Clicking does not run the game's win check; a movement does.
            p = pieces[memory["selected"]]
            s = _current_state(p)
            if p["y"] == 0:
                memory["last_check"] = True
                return ("ACTION1", None, None)
            p["y"] -= 1
            memory["last_check"] = True
            return ("ACTION1", None, None)

        i = memory["selected"] if memory["selected"] in unfinished else unfinished[0]
        if memory["selected"] != i:
            row, col = _click_for_piece(pieces, i)
            memory["selected"] = i
            memory["last_check"] = False
            return ("ACTION6", row, col)

        p = pieces[i]
        tx, ty, option = targets[i]
        current_option = p["stage"] if p["morph"] else p["ori"]
        if current_option != option:
            _rotate_piece(p)
            memory["last_check"] = True
            return ("ACTION5", None, None)
        if p["x"] < tx:
            p["x"] += 1
            action = "ACTION4"
        elif p["x"] > tx:
            p["x"] -= 1
            action = "ACTION3"
        elif p["y"] < ty:
            p["y"] += 1
            action = "ACTION2"
        else:
            p["y"] -= 1
            action = "ACTION1"
        memory["last_check"] = True
        return (action, None, None)

    return ("ACTION1", None, None)

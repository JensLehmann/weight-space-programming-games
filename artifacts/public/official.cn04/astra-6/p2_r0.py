import math
import time
from collections import defaultdict

import numpy as np


# Shapes encode body pixels, ordinary connectors, and special connectors.
# These supply connector-type information when a revealed shape is recognizable;
# unfamiliar shapes are handled by the same geometric solver with unknown types.
_KNOWN_TEXT = (
    ("BBB...", "B.BBBR", "B.....", "B.BBBP", "BBB..."),
    (".BBB", ".B.B", "RB.B", ".B.B", "PB.B", ".B.B", ".BBB"),
    ("PB", ".B", ".B", ".B", ".B", ".P"),
    ("R.....P", "B.....B", "BBB.BBB", "..B.B..", "..BBB.."),
    ("BBBBBB.", "B....BP", "B.BBBB.", "B.B....", "B.B....",
     "BBB....", ".R....."),
    (".BR.", "RB..", ".BBP", ".B.."),
    (".BR.", "RB..", ".BBP", "RB..", ".B.."),
    (".BR...", "RB....", ".BBP..", "RB....", ".BBBBR", ".B...."),
    (".BR...", "RB....", ".BBP..", "RB....", ".BBBBR", "RB....", ".B...."),
    (".BR...", "RB....", ".BBP..", "RB....", ".BBBBR", "RB....", ".BBBP."),
    ("R.R.R", "B.B.B", "BBBBB"),
    ("RBBBB", "....B", ".PBBB"),
    (".RB", "..B", "PBB"),
)


def _point_rot(x, y, w, h, r):
    if r == 0:
        return x, y
    if r == 1:
        return h - 1 - y, x
    if r == 2:
        return w - 1 - x, h - 1 - y
    return y, w - 1 - x


def _rot_shape(s, r):
    w, h, body, pins = s
    b = tuple(sorted(_point_rot(x, y, w, h, r) for x, y in body))
    p = tuple(sorted(_point_rot(x, y, w, h, r) for x, y in pins))
    return (h, w, b, p) if r % 2 else (w, h, b, p)


def _make_known():
    result = []
    for rows in _KNOWN_TEXT:
        body, pins, types = [], [], {}
        for y, row in enumerate(rows):
            for x, c in enumerate(row):
                if c == "B":
                    body.append((x, y))
                elif c in "RP":
                    pins.append((x, y))
                    types[x, y] = 1 if c == "R" else 2
        s = (len(rows[0]), len(rows), tuple(sorted(body)), tuple(sorted(pins)))
        for r in range(4):
            t = _rot_shape(s, r)
            mapping = {
                _point_rot(x, y, s[0], s[1], r): v
                for (x, y), v in types.items()
            }
            result.append((t, tuple(mapping[p] for p in t[3])))
    return tuple(result)


_KNOWN = _make_known()


def _pin_types(shape):
    matches = [t for s, t in _KNOWN if s == shape]
    if matches and all(t == matches[0] for t in matches):
        return matches[0]
    return (0,) * len(shape[3])


def _geometry(obs):
    a = np.asarray(obs)
    bg = int(np.bincount(a[1:].ravel(), minlength=16).argmax())
    divisor = 0
    first_x = first_y = None
    for y in range(1, 64):
        x = 0
        while x < 64:
            c = int(a[y, x])
            e = x + 1
            while e < 64 and int(a[y, e]) == c:
                e += 1
            if c != bg:
                divisor = math.gcd(divisor, e - x)
                if first_x is None:
                    first_x, first_y = x, y
            x = e
    scale = divisor if 1 <= divisor <= 4 else 3
    if scale == 3:
        ox = oy = 2
        width = height = 20
    else:
        ox = (first_x or 0) % scale
        oy = (first_y or 0) % scale
        width = (64 - 2 * ox) // scale
        height = (64 - 2 * oy) // scale
    return bg, scale, ox, oy, width, height


def _grid(obs, m):
    bg, scale, ox, oy, w, h = m["geo"]
    return np.asarray(obs)[
        oy + scale // 2:oy + h * scale:scale,
        ox + scale // 2:ox + w * scale:scale
    ]


def _extract_selected(g, masked, bg):
    h, w = g.shape
    if masked:
        mask = (g != bg) & (g != 4)
        ys, xs = np.where(mask)
        points = {(int(x), int(y)) for x, y in zip(xs, ys)}
    else:
        ys, xs = np.where((g == 0) & (g != bg))
        body = {(int(x), int(y)) for x, y in zip(xs, ys)}
        points = set(body)
        # Connectors of unselected pieces must not be included.
        for x, y in body:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if 0 <= xx < w and 0 <= yy < h and int(g[yy, xx]) in (8, 3):
                    points.add((xx, yy))
    if not points:
        return None
    x0 = min(x for x, y in points)
    y0 = min(y for x, y in points)
    x1 = max(x for x, y in points)
    y1 = max(y for x, y in points)
    body, pins = [], []
    for x, y in points:
        q = (x - x0, y - y0)
        (pins if int(g[y, x]) in (8, 3) else body).append(q)
    s = (x1 - x0 + 1, y1 - y0 + 1,
         tuple(sorted(body)), tuple(sorted(pins)))
    return s, x0, y0


def _new_piece(s, x, y):
    return {
        "shapes": [s], "types": [_pin_types(s)],
        "x": x, "y": y, "v": 0, "r": 0,
        "kind": "new", "edges": {}, "prev": None,
        "history": [0], "reversals": 0,
    }


def _add_variant(o, s):
    try:
        return o["shapes"].index(s)
    except ValueError:
        o["shapes"].append(s)
        o["types"].append(_pin_types(s))
        return len(o["shapes"]) - 1


def _current_shape(o):
    return _rot_shape(o["shapes"][o["v"]], o["r"])


def _occupied(o):
    s = _current_shape(o)
    return {(o["x"] + x, o["y"] + y) for x, y in s[2] + s[3]}


def _group_next(o, v=None, prev=None):
    if v is None:
        v = o["v"]
        prev = o["prev"]
    neighbors = sorted(o["edges"].get(v, ()))
    if not neighbors:
        return v
    onward = [z for z in neighbors if z != prev]
    return onward[0] if onward else neighbors[0]


def _group_cost(o, target):
    v, prev = o["v"], o["prev"]
    for steps in range(2 * len(o["shapes"]) + 3):
        if v == target:
            return steps
        nxt = _group_next(o, v, prev)
        prev, v = v, nxt
    return 100


def _observe_pending(m, selected):
    pending = m.pop("pending", None)
    if pending is None:
        return
    pieces = m["pieces"]
    kind = pending["kind"]

    if kind == "probe":
        i = pending["i"]
        if selected is None:
            pieces[i]["kind"] = "regular"
            m["sel"] = None
            return
        s, x, y = selected
        o = pieces[i]
        o["x"], o["y"] = x, y
        m["sel"] = i
        if o["kind"] == "new":
            if s == _rot_shape(o["shapes"][0], 1):
                o["kind"] = "regular"
                o["r"] = 1
                return
            o["kind"] = "scan"
            v = _add_variant(o, s)
            o["edges"].setdefault(0, set()).add(v)
            o["edges"].setdefault(v, set()).add(0)
            o["history"] = [0, v]
            o["prev"], o["v"], o["r"] = 0, v, 0
            return

        old = o["v"]
        v = _add_variant(o, s)
        hist = o["history"]
        if len(hist) >= 2 and v == hist[-2]:
            o["reversals"] += 1
        o["edges"].setdefault(old, set()).add(v)
        o["edges"].setdefault(v, set()).add(old)
        hist.append(v)
        o["prev"], o["v"], o["r"] = old, v, 0
        if o["reversals"] >= 2 or len(hist) > 35:
            o["kind"] = "group"
        return

    if kind == "group":
        if selected is not None:
            s, x, y = selected
            o = pieces[pending["i"]]
            old = pending["old"]
            v = _add_variant(o, s)
            o["edges"].setdefault(old, set()).add(v)
            o["edges"].setdefault(v, set()).add(old)
            o["prev"], o["v"], o["r"] = old, v, 0
            o["x"], o["y"] = x, y
        return

    if kind == "select":
        if selected is None:
            m["sel"] = None
            return
        s, x, y = selected
        matches = []
        for i, o in enumerate(pieces):
            rotations = range(4) if o["kind"] == "regular" else (0,)
            for v, base in enumerate(o["shapes"]):
                for r in rotations:
                    if _rot_shape(base, r) == s:
                        distance = abs(x - o["x"]) + abs(y - o["y"])
                        preference = 0 if i == pending.get("i") else 1
                        matches.append((distance, preference, i, v, r))
        if matches and min(matches)[0] == 0:
            _, _, i, v, r = min(matches)
            pieces[i].update(x=x, y=y, v=v, r=r)
            m["sel"] = i
            if pending.get("new"):
                m["ignored"].add(pending["cell"])
        elif pending.get("new"):
            pieces.append(_new_piece(s, x, y))
            m["sel"] = len(pieces) - 1
        elif matches:
            _, _, i, v, r = min(matches)
            pieces[i].update(x=x, y=y, v=v, r=r)
            m["sel"] = i
        else:
            m["sel"] = pending.get("i")


def _states(o):
    states = []
    rotations = range(4) if o["kind"] == "regular" else (0,)
    for v, shape in enumerate(o["shapes"]):
        w, h, body, pins = shape
        types = o["types"][v]
        for r in rotations:
            pp = []
            for k, (x, y) in enumerate(pins):
                xx, yy = _point_rot(x, y, w, h, r)
                pp.append((xx, yy, types[k], k))
            rc = ((r - o["r"]) % 4 if o["kind"] == "regular"
                  else _group_cost(o, v))
            states.append({
                "v": v, "r": r,
                "w": h if r % 2 else w,
                "h": w if r % 2 else h,
                "pins": tuple(pp), "rc": rc,
            })
    return states


def _plan(m, seconds=7.3):
    pieces = m["pieces"]
    n = len(pieces)
    if not n:
        return None
    W, H = m["geo"][4:6]
    ss = [_states(o) for o in pieces]
    deadline = time.monotonic() + seconds
    forbidden = m["forbidden"]
    selected = m["sel"]
    salt = m.get("attempt", 0)
    maximum = [max(len(s["pins"]) for s in states) for states in ss]
    parities = [
        {len(s["pins"]) % 2 for s in states} for states in ss
    ]
    anchor = max(range(n), key=lambda i: (maximum[i], i == selected))
    origin_x, origin_y = pieces[anchor]["x"], pieces[anchor]["y"]
    best = [float("inf"), None]
    visits = [0]
    seen = set()
    reach_cache = {0: {(0, 0)}}
    displacement = []
    for states in ss:
        d = set()
        good = True
        for s in states:
            if len(s["pins"]) != 2:
                good = False
                break
            a, b = s["pins"]
            dx, dy = b[0] - a[0], b[1] - a[1]
            d.add((dx, dy))
            d.add((-dx, -dy))
        displacement.append(d if good else None)

    def reach(mask):
        if mask in reach_cache:
            return reach_cache[mask]
        if time.monotonic() > deadline:
            return None
        bit = mask & -mask
        i = bit.bit_length() - 1
        if displacement[i] is None:
            return None
        old = reach(mask ^ bit)
        if old is None:
            return None
        result = {
            (x + dx, y + dy)
            for x, y in old for dx, dy in displacement[i]
        }
        reach_cache[mask] = result
        return result

    def lower_bound(placed):
        xs, ys = [], []
        lo_x = lo_y = -10000
        hi_x = hi_y = 10000
        rc = 0
        for i, (si, x, y) in placed.items():
            s = ss[i][si]
            lo_x = max(lo_x, -x)
            lo_y = max(lo_y, -y)
            hi_x = min(hi_x, W - s["w"] - x)
            hi_y = min(hi_y, H - s["h"] - y)
            xs.append(pieces[i]["x"] - x)
            ys.append(pieces[i]["y"] - y)
            rc += s["rc"]
        if lo_x > hi_x or lo_y > hi_y:
            return None
        xs.sort()
        ys.sort()
        mid = len(xs) // 2
        tx = max(lo_x, min(hi_x, xs[mid]))
        ty = max(lo_y, min(hi_y, ys[mid]))
        cost = rc + sum(abs(x - tx) for x in xs) + sum(abs(y - ty) for y in ys)
        return cost, tx, ty, lo_x, hi_x, lo_y, hi_y, xs, ys

    def finish(placed, pairs, bound):
        signature = tuple(sorted(pairs))
        if signature in forbidden:
            return
        _, tx, ty, lx, hx, ly, hy, xs, ys = bound
        # All median translations have the same movement cost. Prefer ones
        # that leave entire pieces untouched and therefore save selection clicks.
        k = len(xs)
        ax, bx = xs[(k - 1) // 2], xs[k // 2]
        ay, by = ys[(k - 1) // 2], ys[k // 2]
        xchoices = {tx}
        ychoices = {ty}
        for x in xs:
            xchoices.add(max(lx, min(hx, max(ax, min(bx, x)))))
        for y in ys:
            ychoices.add(max(ly, min(hy, max(ay, min(by, y)))))
        for tx in xchoices:
            for ty in ychoices:
                targets = []
                cost = 0
                for i in range(n):
                    si, x, y = placed[i]
                    s = ss[i][si]
                    xx, yy = x + tx, y + ty
                    movement = abs(xx - pieces[i]["x"]) + abs(yy - pieces[i]["y"])
                    change = movement + s["rc"]
                    cost += change + int(change > 0 and i != selected)
                    targets.append((s["v"], s["r"], xx, yy))
                if cost < best[0]:
                    best[0] = cost
                    best[1] = {"targets": targets, "signature": signature}

    def dfs(placed, opened, closed, remaining, pairs, bound):
        visits[0] += 1
        if visits[0] % 128 == 0 and time.monotonic() > deadline:
            raise TimeoutError
        if visits[0] > 160000:
            raise TimeoutError
        if bound[0] >= best[0]:
            return
        key = tuple(sorted((i, *p) for i, p in placed.items()))
        if key in seen:
            return
        seen.add(key)

        if not remaining:
            if not opened:
                finish(placed, pairs, bound)
            return

        ids = [i for i in range(n) if remaining & (1 << i)]
        if len(opened) > sum(maximum[i] for i in ids):
            return
        possible_parity = {0}
        for i in ids:
            possible_parity = {a ^ b for a in possible_parity for b in parities[i]}
        if len(opened) % 2 not in possible_parity:
            return

        if len(opened) == 2 and all(displacement[i] is not None for i in ids):
            positions = list(opened)
            needed = (positions[1][0] - positions[0][0],
                      positions[1][1] - positions[0][1])
            reachable = reach(remaining)
            if reachable is not None and needed not in reachable:
                return

        children = []
        if opened:
            # Any valid final arrangement must cover this still-open connector.
            positions = list(opened)
            q = positions[(len(placed) + salt) % len(positions)]
            endpoint = opened[q]
            placements = []
            for i in ids:
                for si, s in enumerate(ss[i]):
                    for px, py, typ, pin_id in s["pins"]:
                        if endpoint[3] and typ and endpoint[3] != typ:
                            continue
                        placements.append((i, si, q[0] - px, q[1] - py))
        else:
            # A solution may contain several separate connected components.
            i = max(ids, key=lambda j: maximum[j])
            placements = []
            x0 = pieces[i]["x"] - origin_x
            y0 = pieces[i]["y"] - origin_y
            for si in range(len(ss[i])):
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    placements.append((i, si, x0 + dx, y0 + dy))

        unique = set()
        for i, si, x, y in placements:
            identity = (i, si, x, y)
            if identity in unique:
                continue
            unique.add(identity)
            s = ss[i][si]
            op = dict(opened)
            cl = set(closed)
            new_pairs = list(pairs)
            valid = True
            for px, py, typ, pin_id in s["pins"]:
                q = (x + px, y + py)
                if q in cl:
                    valid = False
                    break
                me = (i, s["v"], pin_id, typ)
                if q in op:
                    other = op.pop(q)
                    if typ and other[3] and typ != other[3]:
                        valid = False
                        break
                    cl.add(q)
                    new_pairs.append(tuple(sorted((me[:3], other[:3]))))
                else:
                    op[q] = me
            if not valid:
                continue
            rest = remaining ^ (1 << i)
            if not rest and op:
                continue
            if len(op) > sum(maximum[j] for j in ids if j != i):
                continue
            pl = dict(placed)
            pl[i] = (si, x, y)
            b = lower_bound(pl)
            if b is None or b[0] >= best[0]:
                continue
            # Favor closures when costs tie.
            children.append((b[0], len(op), identity, pl, op, cl, rest, new_pairs, b))
        children.sort(key=lambda z: z[:3])
        for _, _, _, pl, op, cl, rest, new_pairs, b in children:
            dfs(pl, op, cl, rest, new_pairs, b)

    roots = []
    for si, s in enumerate(ss[anchor]):
        placed = {anchor: (si, 0, 0)}
        opened = {
            (x, y): (anchor, s["v"], pin_id, typ)
            for x, y, typ, pin_id in s["pins"]
        }
        b = lower_bound(placed)
        if b is not None:
            roots.append((b[0], si, placed, opened, b))
    roots.sort(key=lambda z: z[:2])
    try:
        for _, si, placed, opened, b in roots:
            dfs(placed, opened, set(), ((1 << n) - 1) ^ (1 << anchor), [], b)
    except TimeoutError:
        pass
    return best[1]


def act(obs, available, memory):
    m = memory
    if "geo" not in m:
        m["geo"] = _geometry(obs)
        g = _grid(obs, m)
        bg = m["geo"][0]
        m["masked"] = bool(np.any((g == 4) & (g != bg)))
        m["pieces"] = []
        m["sel"] = None
        m["ignored"] = set()
        m["forbidden"] = set()
        m["attempt"] = 0
        selected = _extract_selected(g, m["masked"], bg)
        if selected is not None:
            s, x, y = selected
            m["pieces"].append(_new_piece(s, x, y))
            m["sel"] = 0

    g = _grid(obs, m)
    bg, scale, ox, oy, W, H = m["geo"]
    selected = _extract_selected(g, m["masked"], bg)
    previous_action = m.get("last_action")
    _observe_pending(m, selected)
    pieces = m["pieces"]

    def emit(action, row=None, col=None):
        m["last_action"] = action
        return action, row, col

    def click(x, y, i=None, new=False):
        m["pending"] = {
            "kind": "select", "i": i, "new": new, "cell": (x, y)
        }
        return emit("ACTION6", int(oy + y * scale + scale // 2),
                    int(ox + x * scale + scale // 2))

    def move(i, action):
        o = pieces[i]
        dx, dy = {
            "ACTION1": (0, -1), "ACTION2": (0, 1),
            "ACTION3": (-1, 0), "ACTION4": (1, 0)
        }[action]
        s = _current_shape(o)
        x, y = o["x"] + dx, o["y"] + dy
        if 0 <= x <= W - s[0] and 0 <= y <= H - s[1]:
            o["x"], o["y"] = x, y
        return emit(action)

    def rotate_or_cycle(i):
        o = pieces[i]
        if o["kind"] == "regular":
            o["r"] = (o["r"] + 1) % 4
            s = _current_shape(o)
            o["x"] = min(o["x"], max(0, W - s[0]))
            o["y"] = min(o["y"], max(0, H - s[1]))
        else:
            m["pending"] = {"kind": "group", "i": i, "old": o["v"]}
        return emit("ACTION5")

    def unique_cells(i):
        cells = _occupied(pieces[i])
        others = set()
        for j, o in enumerate(pieces):
            if j != i:
                others.update(_occupied(o))
        return cells - others

    def select_piece(i):
        cells = unique_cells(i)
        if cells:
            s = _current_shape(pieces[i])
            body = {(pieces[i]["x"] + x, pieces[i]["y"] + y) for x, y in s[2]}
            preferred = cells & body
            x, y = min(preferred or cells, key=lambda p: (p[1], p[0]))
            return click(x, y, i=i)

        # Temporarily uncover a piece completely hidden under another piece.
        current = m["sel"]
        if current is not None and current != i:
            blocker = pieces[current]
            target = pieces[i]
            bs, ts = _current_shape(blocker), _current_shape(target)
            cx = blocker["x"] + bs[0] / 2 - target["x"] - ts[0] / 2
            cy = blocker["y"] + bs[1] / 2 - target["y"] - ts[1] / 2
            options = []
            for a, dx, dy in (
                ("ACTION1", 0, -1), ("ACTION2", 0, 1),
                ("ACTION3", -1, 0), ("ACTION4", 1, 0)
            ):
                xx, yy = blocker["x"] + dx, blocker["y"] + dy
                if 0 <= xx <= W - bs[0] and 0 <= yy <= H - bs[1]:
                    options.append(((cx + dx) ** 2 + (cy + dy) ** 2, a))
            if options:
                return move(current, max(options)[1])
        cells = _occupied(pieces[i])
        if cells:
            x, y = min(cells)
            return click(x, y, i=i)
        return emit("ACTION1")

    # Identify rotations versus shape-cycling controls.
    if m["sel"] is not None:
        i = m["sel"]
        o = pieces[i]
        if o["kind"] in ("new", "scan"):
            m["pending"] = {"kind": "probe", "i": i}
            return emit("ACTION5")

    if not m.get("learned"):
        # Learn any known but not yet examined piece first.
        for i, o in enumerate(pieces):
            if o["kind"] in ("new", "scan"):
                return select_piece(i)

        covered = set()
        for o in pieces:
            covered.update(_occupied(o))
        candidates = []
        for y in range(H):
            for x in range(W):
                if int(g[y, x]) == bg or (x, y) in covered or (x, y) in m["ignored"]:
                    continue
                neighbors = sum(
                    0 <= x + dx < W and 0 <= y + dy < H
                    and int(g[y + dy, x + dx]) != bg
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
                candidates.append((-neighbors, y, x))
        if candidates:
            _, y, x = min(candidates)
            return click(x, y, new=True)
        m["learned"] = True

    plan = m.get("plan")
    if plan is not None:
        done = all(
            (o["v"], o["r"], o["x"], o["y"]) == tuple(t)
            for o, t in zip(pieces, plan["targets"])
        )
        if done:
            if previous_action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
                # Geometric contact is visible, but a failed completion can
                # reveal that this particular hidden-type pairing is wrong.
                m["forbidden"].add(plan["signature"])
                m["plan"] = None
                m["attempt"] += 1
                plan = None
            else:
                i = m["sel"]
                if i is None:
                    return select_piece(0)
                o = pieces[i]
                s = _current_shape(o)
                if o["y"] == 0:
                    return move(i, "ACTION1")
                if o["y"] + s[1] == H:
                    return move(i, "ACTION2")
                if o["x"] == 0:
                    return move(i, "ACTION3")
                if o["x"] + s[0] == W:
                    return move(i, "ACTION4")
                # The following call restores the piece and checks the goal.
                return move(i, "ACTION1")

    if plan is None:
        plan = _plan(m)
        m["plan"] = plan
        if plan is None:
            m["attempt"] += 1
            # Retry with another search ordering without disturbing the board.
            if m["attempt"] <= 3:
                return emit("ACTION6", 63, 63)
            resets = m.get("resets", 0)
            if "RESET" in available and resets == 0:
                m.clear()
                m["resets"] = resets + 1
                return "RESET", None, None
            # A harmless selection also allows another bounded search pass.
            m["attempt"] = 0
            return emit("ACTION6", 63, 63)

    targets = plan["targets"]
    unfinished = [
        i for i, o in enumerate(pieces)
        if (o["v"], o["r"], o["x"], o["y"]) != tuple(targets[i])
    ]
    if not unfinished:
        i = m["sel"]
        if i is None:
            return select_piece(0)
        o = pieces[i]
        s = _current_shape(o)
        if o["y"] == 0:
            return move(i, "ACTION1")
        if o["x"] == 0:
            return move(i, "ACTION3")
        if o["y"] + s[1] == H:
            return move(i, "ACTION2")
        if o["x"] + s[0] == W:
            return move(i, "ACTION4")
        return move(i, "ACTION1")

    i = m["sel"]
    if i not in unfinished:
        def priority(j):
            o = pieces[j]
            v, r, x, y = targets[j]
            cost = abs(x - o["x"]) + abs(y - o["y"])
            cost += ((r - o["r"]) % 4 if o["kind"] == "regular"
                     else _group_cost(o, v))
            return (not bool(unique_cells(j)), cost, j)
        i = min(unfinished, key=priority)
        return select_piece(i)

    o = pieces[i]
    v, r, x, y = targets[i]
    if o["v"] != v or o["r"] != r:
        return rotate_or_cycle(i)
    if o["x"] < x:
        return move(i, "ACTION4")
    if o["x"] > x:
        return move(i, "ACTION3")
    if o["y"] < y:
        return move(i, "ACTION2")
    return move(i, "ACTION1")

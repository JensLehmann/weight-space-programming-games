import numpy as np
import heapq
import time
from collections import deque


def _components(mask, diagonal=False):
    h, w = mask.shape
    pending = set(np.flatnonzero(mask).tolist())
    result = []
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    while pending:
        root = pending.pop()
        stack, points = [root], [root]
        while stack:
            p = stack.pop()
            y, x = divmod(p, w)
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    q = ny * w + nx
                    if q in pending:
                        pending.remove(q)
                        stack.append(q)
                        points.append(q)
        ys = np.asarray(points, dtype=np.int32) // w
        xs = np.asarray(points, dtype=np.int32) % w
        result.append((
            int(xs.min()), int(ys.min()),
            int(xs.max()) + 1, int(ys.max()) + 1, ys, xs
        ))
    return result


def _empty():
    return np.zeros((64, 64), dtype=bool)


def _point_mask(ys, xs):
    result = _empty()
    ys, xs = np.asarray(ys), np.asarray(xs)
    ok = (ys >= 0) & (ys < 64) & (xs >= 0) & (xs < 64)
    result[ys[ok], xs[ok]] = True
    return result


def _shift(mask, dx, dy):
    ys, xs = np.nonzero(mask)
    return _point_mask(ys + dy, xs + dx)


def _buttons(image, side):
    buttons, controls = {}, {}
    nine, ten = [], []
    for color, name in [(8, "a"), (9, "b"), (6, "c"),
                        (15, "d"), (2, "e"), (11, "f")]:
        mask = image == color
        mask[:, :side] = False
        for x0, y0, x1, y1, ys, xs in _components(mask):
            w, h = x1 - x0, y1 - y0
            k = len(ys) // 2
            point = (int(ys[k]), int(xs[k]))
            if w >= 10 and h >= 4:
                buttons[name] = point
            elif name in ("e", "f") and w >= 5 and h >= 5:
                buttons[name] = point
            elif color == 8 and w >= 7 and 2 <= h <= 3:
                controls["cg"] = point
            elif color == 9 and w == 4 and h == 4:
                nine.append((x0, point))
    mask = image == 10
    mask[:, :side] = False
    for x0, y0, x1, y1, ys, xs in _components(mask):
        if x1 - x0 == 4 and y1 - y0 == 4:
            ten.append((x0, (int(ys[0]), int(xs[0]))))
    nine.sort()
    ten.sort()
    if len(nine) >= 2:
        controls["cu"] = nine[0][1]
        controls["cd"] = nine[-1][1]
    if len(ten) >= 2:
        controls["cl"] = ten[0][1]
        controls["cr"] = ten[-1][1]
    return buttons, controls


def _parse(obs, memory):
    image = np.asarray(obs, dtype=np.int16).copy()
    rows = np.where(np.any(image[:, :25] == 4, axis=1))[0]
    lo, hi = (int(rows[0]), int(rows[-1]) + 1) if len(rows) else (0, 63)
    side = 42
    for x in range(24, 58):
        if np.mean(image[lo:hi, x] == 5) > 0.58:
            side = x
            break

    actor = image == 14
    actor[:, side:] = False
    ay, ax = np.nonzero(actor)
    if not len(ax):
        return None
    px, py = int(ax.min()), int(ay.min())

    goals = []
    mask = image == 11
    mask[:, side:] = False
    for x0, y0, x1, y1, ys, xs in _components(mask):
        if x1 - x0 <= 3 and y1 - y0 <= 3:
            goals.append((x0, y0))
    if goals:
        goal = min(goals, key=lambda p: abs(p[0] - px) + abs(p[1] - py))
        memory["goal"] = goal
    else:
        goal = memory.get("goal")
        if goal is None:
            return None

    buttons, controls = _buttons(image, side)
    previous = memory.get("clean")
    last = memory.get("last", {})
    image[actor] = 4
    restore = previous is not None and last.get("kind") == "move"
    if previous is not None and last.get("kind") == "click":
        action = last.get("group")
        under = previous[py:py + 2, px:px + 2]
        affected = {
            "a": (8, 13), "b": (9,), "c": (6, 7),
            "d": (15,), "e": (1, 12, 5), "f": (1, 12)
        }
        if action in affected and not np.isin(under, affected[action]).any():
            restore = True
    if restore:
        image[actor] = previous[actor]
    else:
        patch = image[max(0, py - 2):min(64, py + 4),
                      max(0, px - 2):min(side, px + 4)]
        colors = [2, 8, 13, 9, 15, 1, 12, 6, 7]
        counts = [int(np.sum(patch == c)) for c in colors]
        color = colors[int(np.argmax(counts))] if max(counts) else 2
        image[actor] = color

    pm = (image == 6) | (image == 7)
    pm[:, side:] = False
    parts = [
        c for c in _components(pm)
        if c[2] - c[0] <= 3 and c[3] - c[1] <= 3
    ]
    hidden = (px, py) in memory.get("portals", [])
    if len(parts) == 1 and "c" in buttons:
        q = parts[0]
        near_red = np.any(
            image[max(0, py - 1):min(64, py + 3),
                  max(0, px - 1):min(side, px + 3)] == 2
        )
        if near_red and (q[0], q[1]) != (px, py):
            hidden = True
    if hidden:
        image[py:py + 2, px:px + 2] = np.array([[6, 7], [7, 6]])

    gx, gy = goal
    image[gy:gy + 2, gx:gx + 2] = 2
    memory["clean"] = image.copy()

    board = _empty()
    board[lo:hi, :side] = True
    floor = ((image == 2) | (image == 1) | (image == 12)) & board
    groups, keys = [], []

    def add(name, variants, start=0, transitions=None):
        groups.append({
            "name": name, "variants": variants,
            "start": start, "transitions": transitions
        })

    blue = ((image == 8) | (image == 13)) & board
    a0, a1 = _empty(), _empty()
    crane_part = None
    for x0, y0, x1, y1, ys, xs in _components(blue):
        center = image[ys, xs] == 13
        if np.any(center):
            cy, cx = ys[center], xs[center]
            anchor_x, anchor_y = int(cx.min()), int(cy.min())
            a0[ys, xs] = True
            if x1 - x0 >= y1 - y0:
                nx = anchor_x + 3 - (ys - anchor_y)
                ny = anchor_y + (xs - anchor_x)
            else:
                nx = anchor_x + (ys - anchor_y)
                ny = anchor_y + 3 - (xs - anchor_x)
            a1 |= _point_mask(ny, nx)
        elif controls and x1 - x0 <= 10 and y1 - y0 <= 10 and len(xs) >= 15:
            if crane_part is None:
                crane_part = (x0, y0, x1, y1, ys, xs)
            else:
                floor[ys, xs] = True
        else:
            floor[ys, xs] = True
    if np.any(a0):
        add("a", [(a0, _empty()), (a1, _empty())])

    for color, name in [(9, "b"), (15, "d")]:
        f0, f1, b0, b1 = _empty(), _empty(), _empty(), _empty()
        count = 0
        for x0, y0, x1, y1, ys, xs in _components(
                (image == color) & board, diagonal=True):
            w, h = x1 - x0, y1 - y0
            if name == "d" and h <= 2 and w <= 6:
                keys.append((x0, y0, x1, y1))
                continue
            if w < 2 or h < 2:
                continue
            count += 1
            solid = len(xs) >= 0.90 * w * h
            yy, xx = np.mgrid[y0:y1, x0:x1]
            checker = ((yy - y0 + xx - x0) % 2) == 1
            if solid:
                f0[y0:y1, x0:x1] = True
                b1[y0:y1, x0:x1] |= checker
            else:
                b0[ys, xs] = True
                f1[y0:y1, x0:x1] = True
        if count:
            add(name, [(f0, b0), (f1, b1)])

    portals = []
    for x0, y0, x1, y1, ys, xs in _components(
            ((image == 6) | (image == 7)) & board):
        floor[ys, xs] = True
        if x1 - x0 <= 3 and y1 - y0 <= 3:
            portals.append((x0, y0))
    memory["portals"] = portals

    bridge_e = None
    for x0, y0, x1, y1, ys, xs in _components((image == 5) & board):
        w, h = x1 - x0, y1 - y0
        if h == 2 and w in (4, 8, 12, 16):
            length = (16 - w) // 2
            bx = x0 - length
            if bx >= 0 and bx + 16 <= side:
                left = image[y0:y1, bx:x0]
                right = image[y0:y1, x1:bx + 16]
                if w == 16 or (
                    left.size and right.size and
                    np.all(left == 1) and np.all(right == 1)
                ):
                    bridge_e = (bx, y0, 16, 2, False)
                    break
        if w == 2 and h in (4, 8, 12, 16):
            length = (16 - h) // 2
            by = y0 - length
            if by >= 0 and by + 16 < 64:
                upper = image[by:y0, x0:x1]
                lower = image[y1:by + 16, x0:x1]
                if h == 16 or (
                    upper.size and lower.size and
                    np.all(upper == 1) and np.all(lower == 1)
                ):
                    bridge_e = (x0, by, 2, 16, True)
                    break
    if bridge_e is None:
        for x0, y0, x1, y1, ys, xs in _components((image == 12) & board):
            if x1 - x0 == 16 and y1 - y0 == 2:
                bridge_e = (x0, y0, 16, 2, False)
                break
            if x1 - x0 == 2 and y1 - y0 == 16:
                bridge_e = (x0, y0, 2, 16, True)
                break

    if bridge_e is not None and ("e" in buttons or memory.get("seen_e")):
        memory["seen_e"] = True
        bx, by, bw, bh, vertical = bridge_e
        patch = image[by:by + bh, bx:bx + bw]
        lengths = [0, 2, 4, 6, 8, 6, 4, 2]
        phase = 4 if np.any(patch == 12) else max(
            0, min(3, int(np.sum(patch == 1) // 8))
        )
        saved = memory.get("e_phase")
        if saved is not None and lengths[saved % 8] == lengths[phase]:
            phase = saved % 8
        variants = []
        for k in range(8):
            p = (phase + k) % 8
            f = _empty()
            length = lengths[p]
            if p == 4:
                f[by:by + bh, bx:bx + bw] = True
            elif length:
                if vertical:
                    f[by:by + length, bx:bx + bw] = True
                    f[by + bh - length:by + bh, bx:bx + bw] = True
                else:
                    f[by:by + bh, bx:bx + length] = True
                    f[by:by + bh, bx + bw - length:bx + bw] = True
            variants.append((f, _empty()))
        floor[by:by + bh, bx:bx + bw] = False
        memory["e_phase"] = phase
        add("e", variants)

    remaining = ((image == 1) | (image == 12)) & board
    if bridge_e is not None:
        bx, by, bw, bh, vertical = bridge_e
        remaining[by:by + bh, bx:bx + bw] = False
    sliding = []
    for x0, y0, x1, y1, ys, xs in _components(remaining):
        w, h = x1 - x0, y1 - y0
        if (w, h) not in ((6, 2), (2, 6)):
            continue
        vertical = h == 6
        endpoint = bool(np.any(image[ys, xs] == 12))
        geometry_key = (vertical, x0 if vertical else y0)
        bases = memory.setdefault("f_bases", {})
        base = bases.get(geometry_key)
        current = y0 if vertical else x0
        if base is None:
            if bridge_e is not None and bridge_e[4] == vertical:
                base = bridge_e[1] if vertical else bridge_e[0]
            elif endpoint:
                base = current - 10
            else:
                if vertical:
                    near = np.argwhere(
                        image[:y0, max(0, x0 - 8):min(side, x1 + 8)] == 2
                    )
                    candidates = [
                        int(p[0]) + 1 for p in near
                        if 0 <= current - (int(p[0]) + 1) <= 10
                    ]
                else:
                    near = np.argwhere(
                        image[max(0, y0 - 8):min(64, y1 + 8), :x0] == 2
                    )
                    candidates = [
                        int(p[1]) + 1 for p in near
                        if 0 <= current - (int(p[1]) + 1) <= 10
                    ]
                candidates = [v for v in candidates if (current - v) % 2 == 0]
                base = max(candidates) if candidates else current
            bases[geometry_key] = base
        phase = 5 if endpoint else max(0, min(5, (current - base) // 2))
        sliding.append((x0, y0, x1, y1, vertical, base, phase))
        floor[ys, xs] = False
    if sliding and ("f" in buttons or memory.get("seen_f")):
        memory["seen_f"] = True
        variants = []
        for k in range(6):
            f = _empty()
            for x0, y0, x1, y1, vertical, base, phase in sliding:
                offset = 2 * ((phase + k) % 6)
                if vertical:
                    f[base + offset:base + offset + 6, x0:x1] = True
                else:
                    f[y0:y1, base + offset:base + offset + 6] = True
            variants.append((f, _empty()))
        add("f", variants)
    else:
        for x0, y0, x1, y1, vertical, base, phase in sliding:
            floor[y0:y1, x0:x1] |= (
                (image[y0:y1, x0:x1] == 1) |
                (image[y0:y1, x0:x1] == 12)
            )

    if crane_part is not None and controls:
        x0, y0, x1, y1, ys, xs = crane_part
        closed = y1 - y0 == 6
        cx, cy = x0, y0 - int(closed)
        obj, center = None, None
        candidates = [
            c for c in _components((image == 12) & board)
            if c[2] - c[0] >= 4 and c[3] - c[1] >= 4
        ]
        if candidates:
            c = min(candidates, key=lambda q:
                    abs((q[0] + q[2]) // 2 - (cx + 9)) +
                    abs((q[1] + q[3]) // 2 - (cy + 4)))
            ox0, oy0, ox1, oy1, oys, oxs = c
            center = ((ox0 + ox1) // 2, (oy0 + oy1) // 2)
            obj = _point_mask(oys, oxs)
            floor[oys, oxs] = False
            saved_shape = memory.get("object_shape")
            if saved_shape is None:
                memory["object_shape"] = (oys - center[1], oxs - center[0])
            else:
                sy, sx = saved_shape
                obj |= _point_mask(sy + center[1], sx + center[0])

        positions = [(0, v) for v in range(4)]
        positions += [(h, 0) for h in range(1, 4)]
        positions += [(h, 3) for h in range(1, 4)]
        legal = set(positions)

        def compatible(bx, by):
            if (cx - bx) % 4 or (by - cy) % 4:
                return False
            return ((cx - bx) // 4, (by - cy) // 4) in legal

        bases = []
        if memory.get("crane_base") is not None:
            bases.append(memory["crane_base"])
        zy, zx = np.nonzero((image == 0) & board)
        if len(zx) >= 20:
            bases.append((int(zx.min()) - 1, int(zy.max()) - 5))
        if center is not None and not closed:
            bases.append((center[0] - 21, center[1] + 8))
        bases.append((cx, cy))
        base_x, base_y = next(
            ((bx, by) for bx, by in bases if compatible(bx, by)), (cx, cy)
        )
        memory["crane_base"] = (base_x, base_y)
        horizontal = (cx - base_x) // 4
        vertical = (base_y - cy) // 4
        attached = closed and center == (cx + 9, cy + 4)

        open_shape = np.array([
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 1, 1],
        ], dtype=bool)
        closed_shape = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)

        states = [(h, v, a) for a in (0, 1) for h, v in positions]
        lookup = {s: i for i, s in enumerate(states)}
        variants, transitions = [], {}
        for i, (h, v, a) in enumerate(states):
            tx, ty = base_x + 4 * h, base_y - 4 * v
            f, b = _empty(), _empty()
            sy, sx = np.nonzero(closed_shape if a else open_shape)
            b |= _point_mask(sy + ty, sx + tx)
            if obj is not None:
                f |= _shift(
                    obj, tx + 9 - center[0], ty + 4 - center[1]
                ) if a else obj
            variants.append((f, b))
            edges = []
            for action, dh, dv in [
                ("cu", 0, 1), ("cd", 0, -1),
                ("cl", -1, 0), ("cr", 1, 0)
            ]:
                target = (h + dh, v + dv, a)
                if action in controls and target in lookup:
                    edges.append((action, lookup[target]))
            if not a and center == (tx + 9, ty + 4) and "cg" in controls:
                edges.append(("cg", lookup[(h, v, 1)]))
            transitions[i] = edges
        start = lookup.get((horizontal, vertical, int(attached)), 0)
        add("z", variants, start, transitions)

    present = {g["name"] for g in groups}
    for name in buttons:
        if name != "c" and name not in present:
            add(name, [(_empty(), _empty())])

    unlocked = "d" in buttons or memory.get("unlocked", False)
    if unlocked:
        memory["unlocked"] = True
    return {
        "image": image, "side": side, "lo": lo, "hi": hi,
        "player": (px, py), "goal": goal, "floor": floor,
        "groups": groups, "buttons": buttons, "controls": controls,
        "portals": portals, "keys": keys, "unlocked": unlocked
    }


def act(obs, available, memory):
    started = time.monotonic()
    available = set(available)

    def simple(name):
        if name in available:
            return name, None, None
        for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"):
            if a in available:
                return a, None, None
        return next(iter(available), "ACTION1"), None, None

    data = _parse(obs, memory)
    if data is None:
        memory.pop("route", None)
        return simple("ACTION1")

    px, py = data["player"]
    gx, gy = data["goal"]
    last = memory.get("last", {})

    def emit(item):
        kind, action = item["kind"], item["action"]
        if item["before"] != data["player"]:
            return None
        if kind == "move":
            if action not in available:
                return None
            memory["last"] = {
                "kind": "move", "position": item["before"],
                "expected": item["after"], "valid": item["valid"]
            }
            return action, None, None
        if "ACTION6" not in available:
            return None
        point = data["controls"].get(action, data["buttons"].get(action))
        if point is None:
            return None
        memory["last"] = {
            "kind": "click", "group": action,
            "group_key": item.get("group_key"),
            "position": item["before"], "expected": item["after"],
            "valid": item["valid"]
        }
        if action == "e" and "e_phase" in memory:
            memory["e_phase"] = (memory["e_phase"] + 1) % 8
        row, col = point
        return "ACTION6", int(row), int(col)

    # Follow a complete route unless its observable prediction fails.
    route = memory.get("route")
    if route:
        expected = last.get("expected")
        if expected is None or expected == data["player"]:
            result = emit(route[0])
            if result is not None:
                route.popleft()
                return result
        memory.pop("route", None)

    if (px, py) == (gx, gy):
        return simple("ACTION1")

    xp, yp = px % 2, py % 2
    xs, ys = np.arange(xp, 63, 2), np.arange(yp, 63, 2)
    W, H = len(xs), len(ys)
    N = W * H
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    def bits(array):
        packed = np.packbits(
            np.asarray(array, dtype=np.uint8).ravel(), bitorder="little"
        )
        return int.from_bytes(packed.tobytes(), "little")

    def masks(f, b):
        return bits(f[yy, xx]), bits(
            b[yy, xx] | b[yy + 1, xx] |
            b[yy, xx + 1] | b[yy + 1, xx + 1]
        )

    def pid(x, y):
        if (x - xp) % 2 or (y - yp) % 2:
            return None
        col, row = (x - xp) // 2, (y - yp) // 2
        return row * W + col if 0 <= col < W and 0 <= row < H else None

    def coords(p):
        return int(xs[p % W]), int(ys[p // W])

    def members(mask):
        while mask:
            low = mask & -mask
            yield low.bit_length() - 1
            mask ^= low

    player, goal = pid(px, py), pid(gx, gy)
    if player is None or goal is None:
        return simple("ACTION1")
    board_bits = bits(
        (xx + 1 < data["side"]) &
        (yy >= data["lo"]) & (yy + 1 < data["hi"])
    )
    base_floor = bits(data["floor"][yy, xx])
    groups = data["groups"]
    for g in groups:
        g["v"] = [masks(f, b) for f, b in g["variants"]]
        g["actual"] = g["v"][g["start"]]

    history = memory.setdefault("history", {})
    previous_group = last.get("group")
    if previous_group in ("a", "b", "d"):
        current = next((g for g in groups if g["name"] == previous_group), None)
        if current is not None:
            old, new = last.get("group_key"), current["actual"]
            if old is not None and old != new:
                learned = history.setdefault(previous_group, {})
                learned[old] = new
                if len(learned) > 12:
                    del learned[next(iter(learned))]

    for g in groups:
        name, count = g["name"], len(g["v"])
        if name == "z":
            g["edges"] = g["transitions"]
            continue
        g["edges"] = {i: [(name, (i + 1) % count)] for i in range(count)}
        if name in ("a", "b", "d"):
            variants = g["v"]
            index = {v: i for i, v in enumerate(variants)}
            for old, new in history.get(name, {}).items():
                for v in (old, new):
                    if v not in index and len(variants) < 12:
                        index[v] = len(variants)
                        variants.append(v)
            for i in range(count, len(variants)):
                g["edges"][i] = [(name, g["start"])]
            for old, new in history.get(name, {}).items():
                if old in index and new in index:
                    g["edges"][index[old]] = [(name, index[new])]

    initial_config, stride = 0, 1
    for g in groups:
        g["stride"] = stride
        initial_config += g["start"] * stride
        stride *= len(g["v"])

    valid_cache = {}

    def valid(config):
        if config in valid_cache:
            return valid_cache[config]
        f, b = base_floor, 0
        for g in groups:
            i = (config // g["stride"]) % len(g["v"])
            gf, gb = g["v"][i]
            f |= gf
            b |= gb
        result = f & ~b & board_bits
        if config == initial_config:
            result |= 1 << player
        valid_cache[config] = result
        return result

    current_valid = valid(initial_config)
    bad = memory.setdefault("bad_edges", set())
    if last.get("kind") == "move":
        old, expected = last.get("position"), last.get("expected")
        if old == data["player"] and expected != data["player"]:
            p, q = pid(*old), pid(*expected)
            if p is not None and q is not None:
                bad.add((last.get("valid", current_valid), p, q))
                bad.add((current_valid, p, q))
    if len(bad) > 400:
        bad = set(list(bad)[-200:])
        memory["bad_edges"] = bad

    neighbours = []
    for p in range(N):
        row, col = divmod(p, W)
        edges = []
        if row:
            edges.append((p - W, "ACTION1"))
        if row + 1 < H:
            edges.append((p + W, "ACTION2"))
        if col:
            edges.append((p - 1, "ACTION3"))
        if col + 1 < W:
            edges.append((p + 1, "ACTION4"))
        neighbours.append([(q, a) for q, a in edges if a in available])

    key_cells = set()
    for x0, y0, x1, y1 in data["keys"]:
        for p in range(N):
            x, y = coords(p)
            if x < x1 and x + 2 > x0 and y < y1 and y + 2 > y0:
                key_cells.add(p)
    unlocked = bool(data["unlocked"] or player in key_cells)

    portal_ids = list(dict.fromkeys(
        q for q in (pid(*p) for p in data["portals"]) if q is not None
    ))
    teleports = {
        p: [q for q in portal_ids if q != p] for p in portal_ids
    } if len(portal_ids) >= 2 else {}

    def md(p, q):
        return abs(p % W - q % W) + abs(p // W - q // W)

    heuristic = [md(p, goal) for p in range(N)]
    if "c" in data["buttons"]:
        for p in range(N):
            for a, targets in teleports.items():
                for b in targets:
                    heuristic[p] = min(
                        heuristic[p], md(p, a) + 1 + md(b, goal)
                    )

    def permitted(action, flag):
        if action in ("cu", "cd", "cl", "cr", "cg"):
            return action in data["controls"]
        if action == "d":
            return flag and ("d" in data["buttons"] or bool(data["keys"]))
        return action in data["buttons"]

    all_bits = (1 << N) - 1
    not_left = all_bits ^ sum(1 << (r * W) for r in range(H))
    not_right = all_bits ^ sum(1 << (r * W + W - 1) for r in range(H))
    component_cache = {}
    bad_by_mask = {}
    for mask, p, q in bad:
        bad_by_mask.setdefault(mask, set()).add((p, q))
    unrestricted = all(
        a in available for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
    )

    def regions(mask):
        if mask in component_cache:
            return component_cache[mask]
        remaining, result = mask, []
        forbidden = bad_by_mask.get(mask)
        if not forbidden and unrestricted:
            while remaining:
                region = remaining & -remaining
                while True:
                    spread = (
                        (region << W) | (region >> W) |
                        ((region & not_left) >> 1) |
                        ((region & not_right) << 1)
                    )
                    new = region | (spread & mask)
                    if new == region:
                        break
                    region = new
                result.append(region)
                remaining &= ~region
        else:
            forbidden = forbidden or set()
            while remaining:
                root = (remaining & -remaining).bit_length() - 1
                queue = [root]
                region = 1 << root
                remaining &= ~region
                while queue:
                    p = queue.pop()
                    for q, action in neighbours[p]:
                        if remaining & (1 << q) and (p, q) not in forbidden:
                            remaining &= ~(1 << q)
                            region |= 1 << q
                            queue.append(q)
                result.append(region)
        component_cache[mask] = result
        return result

    def containing(mask, p):
        for region in regions(mask):
            if region & (1 << p):
                return region
        return 1 << p

    def walk_info(mask, source):
        distance, predecessor, first = {source: 0}, {source: None}, {}
        queue = deque([source])
        forbidden = bad_by_mask.get(mask, ())
        while queue:
            p = queue.popleft()
            for q, action in neighbours[p]:
                if q in distance or not (mask & (1 << q)):
                    continue
                if (p, q) in forbidden:
                    continue
                distance[q] = distance[p] + 1
                predecessor[q] = (p, action)
                first[q] = action if p == source else first[p]
                queue.append(q)
        return distance, predecessor, first

    initial_region = containing(current_valid, player)
    initial_state = (initial_config, initial_region, int(unlocked))
    costs = {initial_state: 0}
    entries = {initial_state: player}
    parents = {}
    heap = [(heuristic[player], 0, 0, initial_state)]
    serial, expansions, solved = 1, 0, None

    while heap and expansions < 50000:
        if expansions % 64 == 0 and time.monotonic() - started > 6.5:
            break
        _, cost, _, state = heapq.heappop(heap)
        if costs.get(state) != cost:
            continue
        config, region, flag = state
        entry = entries[state]
        mask = valid(config)
        wd, wp, wf = walk_info(mask, entry)
        if goal in wd:
            solved = state
            break
        expansions += 1

        def offer(nc, nr, nf, np_, extra_cost, macro):
            nonlocal serial
            ns = (nc, nr, int(nf))
            new_cost = cost + extra_cost
            if new_cost >= costs.get(ns, 10**9):
                return
            costs[ns], entries[ns] = new_cost, np_
            parents[ns] = (state, macro)
            heapq.heappush(
                heap, (new_cost + 1.1 * heuristic[np_], new_cost, serial, ns)
            )
            serial += 1

        if not flag:
            possible = [p for p in key_cells if p in wd]
            if possible:
                p = min(possible, key=lambda q: wd[q])
                offer(config, region, True, p, wd[p], ("walk", None, p))

        if "ACTION6" not in available:
            continue
        for g in groups:
            i = (config // g["stride"]) % len(g["v"])
            for action, j in g["edges"].get(i, []):
                if i == j or not permitted(action, flag):
                    continue
                nc = config + (j - i) * g["stride"]
                nv = valid(nc)
                survivors = region & nv
                if not survivors:
                    continue
                for nr in regions(nv):
                    intersection = survivors & nr
                    if not intersection:
                        continue
                    possible = [p for p in members(intersection) if p in wd]
                    if not possible:
                        continue
                    p = min(possible, key=lambda q: wd[q] + .08 * heuristic[q])
                    offer(nc, nr, flag or p in key_cells, p, wd[p] + 1,
                          ("click", action, p))
        if "c" in data["buttons"]:
            for p, destinations in teleports.items():
                if p not in wd:
                    continue
                for q in destinations:
                    if mask & (1 << q):
                        offer(config, containing(mask, q),
                              flag or q in key_cells, q, wd[p] + 1,
                              ("click", "c", p))

    # Expand the entire solution into a committed, position-checked route.
    if solved is not None:
        chain = []
        state = solved
        while state != initial_state:
            previous, macro = parents[state]
            chain.append((previous, state, macro))
            state = previous
        chain.reverse()
        route = deque()

        def append_walk(config, source, target):
            mask = valid(config)
            wd, pred, first = walk_info(mask, source)
            if target not in wd:
                return False
            steps = []
            q = target
            while q != source:
                p, action = pred[q]
                steps.append((p, q, action))
                q = p
            for p, q, action in reversed(steps):
                route.append({
                    "kind": "move", "action": action,
                    "before": coords(p), "after": coords(q), "valid": mask
                })
            return True

        good = True
        for source_state, target_state, macro in chain:
            kind, action, target = macro
            config = source_state[0]
            if not append_walk(config, entries[source_state], target):
                good = False
                break
            if kind == "click":
                g = next((g for g in groups if g["name"] == action), None)
                group_key = None
                if g is not None:
                    index = (config // g["stride"]) % len(g["v"])
                    group_key = g["v"][index]
                route.append({
                    "kind": "click", "action": action,
                    "before": coords(target),
                    "after": coords(entries[target_state]),
                    "valid": valid(config), "group_key": group_key
                })
        if good:
            good = append_walk(solved[0], entries[solved], goal)
        if good and route:
            result = emit(route[0])
            if result is not None:
                route.popleft()
                memory["route"] = route
                return result

    # Empirical fallback: prefer unseen safe positions and control states.
    wd, wp, wf = walk_info(current_valid, player)
    visits = memory.setdefault("visits", {})
    vkey = (current_valid, player)
    visits[vkey] = visits.get(vkey, 0) + 1
    probes = memory.setdefault("probes", {})
    candidates = []

    if goal in wf:
        chosen = ("move", wf[goal])
    else:
        chosen = None
        if "ACTION6" in available:
            for g in groups:
                i = g["start"]
                edges = g["edges"].get(i, [])
                if len(g["v"]) == 1:
                    edges = [(g["name"], i)]
                for action, j in edges:
                    if not permitted(action, unlocked):
                        continue
                    nc = initial_config + (j - i) * g["stride"]
                    nv = valid(nc)
                    suitable = [
                        p for p in wd
                        if (nv & (1 << p)) or (base_floor & (1 << p))
                    ]
                    if not suitable:
                        continue
                    target = min(suitable, key=lambda p: wd[p])
                    signature = (action, g["actual"], current_valid)
                    score = wd[target] + 15 * probes.get(signature, 0)
                    if g["name"] == "z":
                        score += 3
                    candidates.append((score, target, action, signature))
            if "c" in data["buttons"]:
                for p in portal_ids:
                    if p in wd:
                        signature = ("c", p, current_valid)
                        score = wd[p] + 15 * probes.get(signature, 0)
                        candidates.append((score, p, "c", signature))

        for p in wd:
            if p == player:
                continue
            seen = visits.get((current_valid, p), 0)
            score = wd[p] + .30 * heuristic[p] + 12 * seen + 2
            if p in key_cells and not unlocked:
                score -= 15
            candidates.append((score, p, None, None))
        if candidates:
            _, target, action, signature = min(
                candidates, key=lambda t: (t[0], wd[t[1]])
            )
            if target != player:
                chosen = ("move", wf[target])
            elif action is not None:
                chosen = ("click", action)
                probes[signature] = probes.get(signature, 0) + 1

    if chosen is None:
        for q, action in neighbours[player]:
            if q in wd:
                chosen = ("move", action)
                break
    if chosen is None:
        return simple("ACTION1")

    kind, action = chosen
    if kind == "move":
        dx, dy = {
            "ACTION1": (0, -2), "ACTION2": (0, 2),
            "ACTION3": (-2, 0), "ACTION4": (2, 0)
        }[action]
        item = {
            "kind": kind, "action": action, "before": (px, py),
            "after": (px + dx, py + dy), "valid": current_valid
        }
    else:
        g = next((g for g in groups if g["name"] == action), None)
        expected = (px, py)
        if action == "c" and player in teleports:
            expected = coords(teleports[player][0])
        item = {
            "kind": kind, "action": action, "before": (px, py),
            "after": expected, "valid": current_valid,
            "group_key": g["actual"] if g is not None else None
        }
    result = emit(item)
    return result if result is not None else simple("ACTION1")

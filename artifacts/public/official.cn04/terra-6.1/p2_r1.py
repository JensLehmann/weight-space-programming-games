from collections import deque
from math import gcd
import numpy as np


def _components(mask):
    points = set(map(tuple, np.argwhere(mask)))
    result = []
    while points:
        y, x = points.pop()
        stack = [(y, x)]
        component = []
        while stack:
            cy, cx = stack.pop()
            component.append((cx, cy))
            for q in ((cy - 1, cx), (cy + 1, cx),
                      (cy, cx - 1), (cy, cx + 1)):
                if q in points:
                    points.remove(q)
                    stack.append(q)
        result.append(component)
    return result


def _camera(obs):
    a = np.asarray(obs)
    xs = set()
    ys = set()
    for y in range(1, 64):
        for x in range(1, 64):
            if a[y, x] != a[y, x - 1]:
                xs.add(x)
            if y >= 2 and a[y, x] != a[y - 1, x]:
                ys.add(y)

    def axis_gcd(values):
        values = sorted(values)
        if len(values) < 2:
            return 0
        g = 0
        for v in values[1:]:
            g = gcd(g, v - values[0])
        return g

    gx = axis_gcd(xs)
    gy = axis_gcd(ys)
    scale = gcd(gx, gy) if gx and gy else (gx or gy)
    if scale < 2:
        scale = 3
    if scale > 8:
        divisors = [d for d in range(2, 9) if scale % d == 0]
        scale = max(divisors) if divisors else 3

    def dimension(values):
        residue = min(values) % scale if values else 0
        candidates = []
        for n in range(1, 65):
            if 64 // n != scale:
                continue
            offset = (64 - n * scale) // 2
            if offset % scale == residue:
                candidates.append((n, offset))
        if candidates:
            return max(candidates)
        n = 64 // scale
        return n, (64 - n * scale) // 2

    w, ox = dimension(xs)
    h, oy = dimension(ys)
    return scale, ox, oy, w, h


def _read_grid(obs, m):
    s, ox, oy, w, h = m["camera"]
    a = np.asarray(obs)
    return np.array([
        [int(a[oy + y * s + s // 2, ox + x * s + s // 2])
         for x in range(w)]
        for y in range(h)
    ], dtype=int)


def _piece(points):
    x0 = min(x for x, y in points)
    y0 = min(y for x, y in points)
    cells = sorted((x - x0, y - y0) for x, y in points)
    return {
        "x": x0, "y": y0,
        "w": max(x for x, y in cells) + 1,
        "h": max(y for x, y in cells) + 1,
        "cells": cells,
        "ep": [],
        "r": 0,
        "marker": False,
        "group": False,
    }


def _rotated(points, w, h, r):
    pts = list(points)
    for _ in range(r % 4):
        pts = [(h - 1 - y, x) for x, y in pts]
        w, h = h, w
    return pts, w, h


def _absolute_cells(p):
    cells, _, _ = _rotated(p["cells"], p["w"], p["h"], p["r"])
    return {(p["x"] + x, p["y"] + y) for x, y in cells}


def _record(m, index, grid):
    """Read the currently selected piece."""
    p = m["pieces"][index]
    bg = m["background"]

    if m["grey"]:
        active = (grid != bg) & (grid != 4)
        comps = _components(active)
        old = _absolute_cells(p)
        best = None
        best_score = -1
        for comp in comps:
            overlap = sum(pt in old for pt in comp)
            if overlap:
                score = overlap * 100 + len(comp)
                if score > best_score:
                    best_score = score
                    best = comp
        if best:
            # When the old component is mostly the selected piece, retain
            # its grey cells too; otherwise use the highlighted component.
            old_points = list(old)
            if len(best) >= 0.6 * len(old_points):
                points = set(old_points) | set(best)
            else:
                points = set(best)
            p2 = _piece(list(points))
            p["x"], p["y"] = p2["x"], p2["y"]
            p["w"], p["h"] = p2["w"], p2["h"]
            p["cells"] = p2["cells"]
            p["r"] = 0

    endpoints = []
    marker = False
    for x, y in p["cells"]:
        gx, gy = p["x"] + x, p["y"] + y
        if 0 <= gy < grid.shape[0] and 0 <= gx < grid.shape[1]:
            color = int(grid[gy, gx])
            if color in (8, 3):
                endpoints.append((x, y))
            if m["grey"] and color == 0:
                marker = True
    p["ep"] = sorted(set(endpoints))
    p["marker"] = marker
    p["r"] = 0


def _click(m, x, y):
    s, ox, oy, _, _ = m["camera"]
    return ("ACTION6", oy + y * s + s // 2, ox + x * s + s // 2)


def _click_piece(m, index):
    pieces = m["pieces"]
    mine = _absolute_cells(pieces[index])
    others = set()
    for j, p in enumerate(pieces):
        if j != index:
            others.update(_absolute_cells(p))
    choices = sorted(mine - others, key=lambda q: (q[1], q[0]))
    if not choices:
        choices = sorted(mine, key=lambda q: (q[1], q[0]))
    if not choices:
        return ("ACTION1", None, None)
    return _click(m, *choices[0])


def _initialize(obs, memory):
    camera = _camera(obs)
    m = {
        "camera": camera,
        "phase": "scan",
        "selected": -1,
        "pending": -1,
        "scan_index": 0,
        "tried": [],
        "target": None,
        "target_moved": False,
        "solutions": [],
        "fallback": 0,
    }
    grid = _read_grid(obs, m)
    counts = np.bincount(grid.ravel(), minlength=16)
    bg = int(np.argmax(counts))
    m["background"] = bg

    foreground = grid != bg
    grey = bg != 4 and int(np.count_nonzero(grid == 4)) >= 3
    m["grey"] = grey

    if grey:
        active = foreground & (grid != 4)
        inactive = foreground & (grid == 4)
        active_comps = _components(active)
        inactive_comps = _components(inactive)
        comps = active_comps + inactive_comps
        selected = -1
        best = 0
        for i, comp in enumerate(active_comps):
            score = sum(int(grid[y, x] == 8) * 10 + 1 for x, y in comp)
            if score > best:
                best = score
                selected = i
    else:
        comps = _components(foreground)
        selected = -1
        best = 0
        for i, comp in enumerate(comps):
            score = sum(int(grid[y, x] == 0) for x, y in comp)
            if score > best:
                best = score
                selected = i

    comps = [c for c in comps if c]
    m["pieces"] = [_piece(c) for c in comps]
    m["selected"] = selected if selected < len(comps) else -1
    m["scanned"] = [False] * len(comps)
    if m["selected"] >= 0:
        _record(m, m["selected"], grid)
        m["scanned"][m["selected"]] = True

    memory.clear()
    memory.update(m)


def _pose_cost(p, pose):
    x, y, r = pose
    return abs(x - p["x"]) + abs(y - p["y"]) + (r - p["r"]) % 4


def _pair_signature(pieces, poses):
    locations = {}
    for i, (p, pose) in enumerate(zip(pieces, poses)):
        x, y, r = pose
        ep, _, _ = _rotated(p["ep"], p["w"], p["h"], r)
        for j, (dx, dy) in enumerate(ep):
            locations.setdefault((x + dx, y + dy), []).append((i, j))
    pairs = []
    for entries in locations.values():
        if len(entries) != 2:
            return None
        pairs.append(tuple(sorted(entries)))
    return repr(tuple(sorted(pairs)))


def _search(m):
    pieces = m["pieces"]
    n = len(pieces)
    if not n or any(not p["ep"] for p in pieces):
        return []
    if sum(len(p["ep"]) for p in pieces) % 2:
        return []

    _, _, _, width, height = m["camera"]
    rotations = []
    for p in pieces:
        options = []
        for r in range(4):
            ep, w, h = _rotated(p["ep"], p["w"], p["h"], r)
            options.append((ep, w, h))
        rotations.append(options)

    ranked = sorted(
        range(n),
        key=lambda i: (pieces[i]["group"], len(pieces[i]["ep"])),
        reverse=True,
    )
    if any(p["group"] for p in pieces):
        anchors = [i for i in ranked if pieces[i]["group"]][:2]
    else:
        anchors = ranked[:min(3, n)]

    finals = []
    beam_limit = 220

    def run(anchor, anchor_rotations):
        p = pieces[anchor]
        for ar in anchor_rotations:
            if p["group"] and ar != 0:
                continue
            ep, aw, ah = rotations[anchor][ar]
            ax = min(p["x"], max(0, width - aw))
            ay = min(p["y"], max(0, height - ah))
            if ax < 0 or ay < 0 or ax + aw > width or ay + ah > height:
                continue
            poses = [None] * n
            poses[anchor] = (ax, ay, ar)
            occ = {}
            for ex, ey in ep:
                q = (ax + ex, ay + ey)
                occ[q] = occ.get(q, 0) + 1
            if max(occ.values(), default=0) > 2:
                continue
            beam = [(_pose_cost(p, poses[anchor]), poses, occ)]

            for depth in range(1, n):
                candidates = {}
                for cost, state, occupied in beam:
                    open_points = [q for q, count in occupied.items()
                                   if count == 1]
                    if not open_points:
                        continue
                    for i in range(n):
                        if state[i] is not None:
                            continue
                        pi = pieces[i]
                        rs = (0,) if pi["group"] else range(4)
                        generated = set()
                        for r in rs:
                            eps, rw, rh = rotations[i][r]
                            for ox, oy in open_points:
                                for ex, ey in eps:
                                    x, y = ox - ex, oy - ey
                                    pose = (x, y, r)
                                    if pose in generated:
                                        continue
                                    generated.add(pose)
                                    if (x < 0 or y < 0 or
                                            x + rw > width or y + rh > height):
                                        continue
                                    new_occ = occupied.copy()
                                    valid = True
                                    for dx, dy in eps:
                                        q = (x + dx, y + dy)
                                        value = new_occ.get(q, 0) + 1
                                        if value > 2:
                                            valid = False
                                            break
                                        new_occ[q] = value
                                    if not valid:
                                        continue

                                    new_state = state.copy()
                                    new_state[i] = pose
                                    remaining = sum(
                                        len(pieces[j]["ep"])
                                        for j in range(n)
                                        if new_state[j] is None
                                    )
                                    unmatched = sum(
                                        v == 1 for v in new_occ.values()
                                    )
                                    if unmatched > remaining:
                                        continue
                                    if (remaining - unmatched) % 2:
                                        continue

                                    new_cost = cost + _pose_cost(pi, pose)
                                    key = tuple(new_state)
                                    score = new_cost + 2 * unmatched
                                    previous = candidates.get(key)
                                    if previous is None or score < previous[0]:
                                        candidates[key] = (
                                            score, new_cost, new_state, new_occ
                                        )
                if not candidates:
                    beam = []
                    break
                best = sorted(candidates.values(), key=lambda v: v[0])
                beam = [(c, st, oc)
                        for _, c, st, oc in best[:beam_limit]]

            for cost, state, occupied in beam:
                if all(v == 2 for v in occupied.values()):
                    sig = _pair_signature(pieces, state)
                    if sig is not None:
                        finals.append((cost, sig, state))

    for anchor in anchors:
        run(anchor, (0,))

    if not finals:
        for anchor in anchors:
            run(anchor, (1, 2, 3))

    finals.sort(key=lambda v: v[0])
    # Keep several geometries per pairing, but favor different pairings.
    per_signature = {}
    for cost, sig, state in finals:
        per_signature.setdefault(sig, [])
        if len(per_signature[sig]) < 3:
            per_signature[sig].append(state)
    solutions = []
    for sig, states in per_signature.items():
        for state in states:
            solutions.append({"signature": sig, "poses": state})
        if len(solutions) >= 120:
            break
    return solutions


def _next_solution(m):
    tried = set(m["tried"])
    candidates = []
    for solution in m["solutions"]:
        if solution["signature"] in tried:
            continue
        poses = solution["poses"]
        cost = sum(_pose_cost(p, pose)
                   for p, pose in zip(m["pieces"], poses))
        switches = sum(
            tuple(pose) != (p["x"], p["y"], p["r"])
            for p, pose in zip(m["pieces"], poses)
        )
        candidates.append((cost + switches, solution))
    if not candidates:
        return False
    _, solution = min(candidates, key=lambda v: v[0])
    m["target"] = solution["poses"]
    m["target_moved"] = False
    m["tried"].append(solution["signature"])
    return True


def act(obs, available, memory):
    if "phase" not in memory:
        _initialize(obs, memory)
    m = memory
    grid = _read_grid(obs, m)

    # Finish a selection or a shape-changing action from the previous call.
    if m["pending"] >= 0:
        i = m["pending"]
        _record(m, i, grid)
        m["selected"] = i
        if m["phase"] == "scan":
            m["scanned"][i] = True
        m["pending"] = -1

    while True:
        if m["phase"] == "scan":
            unscanned = [i for i, done in enumerate(m["scanned"]) if not done]
            if unscanned:
                i = unscanned[0]
                m["pending"] = i
                return _click_piece(m, i)

            # A zero marker on a highlighted, grey-masked piece indicates
            # the shape-cycling pieces used by this game.
            for p in m["pieces"]:
                if m["grey"] and p["marker"]:
                    p["group"] = True
            m["phase"] = "expand"
            continue

        if m["phase"] == "expand":
            pieces = m["pieces"]
            candidate = None
            for i, p in enumerate(pieces):
                if not p["group"] or not p["marker"]:
                    continue
                other = sum(len(q["ep"]) for j, q in enumerate(pieces)
                            if j != i)
                if len(p["ep"]) < other:
                    candidate = i
                    break
            if candidate is None:
                m["phase"] = "plan"
                continue
            if m["selected"] != candidate:
                m["selected"] = candidate
                m["pending"] = candidate
                return _click_piece(m, candidate)
            m["pending"] = candidate
            return ("ACTION5", None, None)

        if m["phase"] == "plan":
            m["solutions"] = _search(m)
            m["phase"] = "execute"
            if not _next_solution(m):
                m["phase"] = "fallback"
            continue

        if m["phase"] == "execute":
            target = m["target"]
            if target is None:
                if not _next_solution(m):
                    m["phase"] = "fallback"
                    continue
                target = m["target"]

            changed = [
                i for i, (p, pose) in enumerate(zip(m["pieces"], target))
                if (p["x"], p["y"], p["r"]) != tuple(pose)
            ]
            if not changed:
                if not m["target_moved"] and m["selected"] >= 0:
                    # Completion is checked on a move or rotation, not a click.
                    i = m["selected"]
                    p = m["pieces"][i]
                    _, w, h = _rotated(p["cells"], p["w"], p["h"], p["r"])
                    _, _, _, gw, gh = m["camera"]
                    if p["x"] + w < gw:
                        p["x"] += 1
                        m["target_moved"] = True
                        return ("ACTION4", None, None)
                    if p["x"] > 0:
                        p["x"] -= 1
                        m["target_moved"] = True
                        return ("ACTION3", None, None)
                m["target"] = None
                continue

            if m["selected"] in changed:
                i = m["selected"]
            else:
                i = min(changed, key=lambda j: _pose_cost(
                    m["pieces"][j], target[j]))
                m["selected"] = i
                return _click_piece(m, i)

            p = m["pieces"][i]
            tx, ty, tr = target[i]
            if p["r"] != tr and not p["group"]:
                p["r"] = (p["r"] + 1) % 4
                _, w, h = _rotated(p["cells"], p["w"], p["h"], p["r"])
                _, _, _, gw, gh = m["camera"]
                p["x"] = min(p["x"], max(0, gw - w))
                p["y"] = min(p["y"], max(0, gh - h))
                m["target_moved"] = True
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
            elif p["y"] > ty:
                p["y"] -= 1
                action = "ACTION1"
            else:
                m["target"] = None
                continue
            m["target_moved"] = True
            return (action, None, None)

        # If parsing or search fails, restart once and rescan.
        if m["fallback"] == 0 and "RESET" in available:
            m["fallback"] = 1
            memory.clear()
            return ("RESET", None, None)
        return ("ACTION1", None, None)

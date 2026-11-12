from collections import deque
from itertools import product


def _geometry(obs):
    # The usual viewport is 21x21 at scale 3. Also consider the other
    # common square viewport sizes used by the engine.
    choices = []
    for scale, n in ((3, 21), (4, 16), (5, 12)):
        pad = (64 - scale * n) // 2
        bad = 0
        total = 0
        for y in range(n):
            yy = pad + y * scale
            for x in range(n):
                xx = pad + x * scale
                if xx + scale - 1 >= 63:
                    continue
                corners = (
                    int(obs[yy, xx]),
                    int(obs[yy, xx + scale - 1]),
                    int(obs[yy + scale - 1, xx]),
                    int(obs[yy + scale - 1, xx + scale - 1]),
                )
                bad += 4 - max(corners.count(c) for c in set(corners))
                total += 4
        choices.append((bad / max(1, total), scale, n, pad))
    choices.sort()
    _, scale, n, pad = choices[0]
    return scale, n, pad


def _read(obs):
    scale, n, pad = _geometry(obs)
    marker = 2 if scale == 5 else 1
    base = [[0] * n for _ in range(n)]
    mid = [[0] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            yy, xx = pad + y * scale, pad + x * scale
            base[y][x] = int(obs[yy, xx])
            mid[y][x] = int(obs[yy + marker, xx + marker])

    targets = []
    for y in range(n):
        for x in range(n):
            if base[y][x] == 11 or mid[y][x] == 11:
                targets.append((x, y))

    # Mirrors are long color-10 lines. Their center marks distinguish
    # movable mirrors from fixed ones.
    axes = []
    threshold = max(5, n // 3)
    for x in range(n):
        cells = [(x, y) for y in range(n) if base[y][x] == 10]
        if len(cells) >= threshold:
            marks = [mid[y][x] for x, y in cells]
            axes.append({
                "kind": "v",
                "p": x,
                "movable": any(c in (0, 9) for c in marks),
                "selected": 0 in marks,
            })
    for y in range(n):
        cells = [(x, y) for x in range(n) if base[y][x] == 10]
        if len(cells) >= threshold:
            marks = [mid[y][x] for x, y in cells]
            axes.append({
                "kind": "h",
                "p": y,
                "movable": any(c in (0, 9) for c in marks),
                "selected": 0 in marks,
            })

    # Original pieces are color 5; reflected copies are color 4.
    original = {(x, y) for y in range(n) for x in range(n)
                if base[y][x] == 5}
    remaining = set(original)
    pieces = []
    while remaining:
        seed = min(remaining, key=lambda p: (p[1], p[0]))
        remaining.remove(seed)
        q = deque([seed])
        component = [seed]
        while q:
            x, y = q.popleft()
            for pt in ((x - 1, y), (x + 1, y),
                       (x, y - 1), (x, y + 1)):
                if pt in remaining:
                    remaining.remove(pt)
                    q.append(pt)
                    component.append(pt)
        x0 = min(x for x, y in component)
        y0 = min(y for x, y in component)
        offsets = sorted((x - x0, y - y0) for x, y in component)
        marks = [mid[y][x] for x, y in component]
        pieces.append({
            "x": x0,
            "y": y0,
            "cells": offsets,
            "w": max(dx for dx, dy in offsets) + 1,
            "h": max(dy for dx, dy in offsets) + 1,
            "movable": not (5 in marks and 0 not in marks and 9 not in marks),
            "selected": 0 in marks,
            "allow_v": True,
            "allow_h": True,
        })

    # Some pieces reflect in only one direction. A visibly absent direct
    # reflection is evidence that the corresponding direction is disabled.
    for piece in pieces:
        own = {(piece["x"] + dx, piece["y"] + dy)
               for dx, dy in piece["cells"]}
        for axis in axes:
            absent = False
            for x, y in own:
                if axis["kind"] == "v":
                    rx, ry = 2 * axis["p"] - x, y
                else:
                    rx, ry = x, 2 * axis["p"] - y
                if not (0 <= rx < n and 0 <= ry < n):
                    continue
                if (rx, ry) in original:
                    continue
                if base[ry][rx] != 4:
                    absent = True
                    break
            if absent:
                piece["allow_" + axis["kind"]] = False

    selected = None
    for i, axis in enumerate(axes):
        if axis["selected"]:
            selected = ("a", i)
            break
    if selected is None:
        for i, piece in enumerate(pieces):
            if piece["selected"]:
                selected = ("p", i)
                break
    if selected is None:
        for i, axis in enumerate(axes):
            if axis["movable"]:
                selected = ("a", i)
                break
    if selected is None:
        for i, piece in enumerate(pieces):
            if piece["movable"]:
                selected = ("p", i)
                break

    return {
        "scale": scale,
        "n": n,
        "pad": pad,
        "base": base,
        "mid": mid,
        "targets": targets,
        "axes": axes,
        "pieces": pieces,
        "selected": selected,
    }


def _orbit(x, y, axes, positions, piece):
    allowed = []
    for axis, p in zip(axes, positions):
        if piece["allow_" + axis["kind"]]:
            allowed.append((axis["kind"], p))

    # One mirror per direction is by far the most common case.
    if len(allowed) <= 2 and len({k for k, p in allowed}) == len(allowed):
        result = {(x, y)}
        for kind, p in allowed:
            if kind == "v":
                result |= {(2 * p - a, b) for a, b in tuple(result)}
            else:
                result |= {(a, 2 * p - b) for a, b in tuple(result)}
        return result

    seen = {(x, y)}
    q = deque([(x, y, 0)])
    while q:
        a, b, depth = q.popleft()
        if depth >= 13:
            continue
        for kind, p in allowed:
            pt = (2 * p - a, b) if kind == "v" else (a, 2 * p - b)
            if pt not in seen:
                seen.add(pt)
                q.append((pt[0], pt[1], depth + 1))
    return seen


def _solve(state):
    n = state["n"]
    targets = state["targets"]
    axes = state["axes"]
    pieces = state["pieces"]
    if not targets or not pieces:
        return None

    full = (1 << len(targets)) - 1
    current_axes = tuple(a["p"] for a in axes)
    current_positions = [(p["x"], p["y"]) for p in pieces]

    ranges = [range(n) if a["movable"] else (a["p"],) for a in axes]
    axis_options = list(product(*ranges)) if ranges else [()]
    axis_options.sort(key=lambda ps: sum(abs(p - a["p"])
                                         for p, a in zip(ps, axes)))

    initial_covered = 0
    for i, (x, y) in enumerate(targets):
        if state["base"][y][x] in (4, 5):
            initial_covered |= 1 << i
    initial_count = initial_covered.bit_count()

    best_cost = [10 ** 9]
    best = [None]
    partial_count = [initial_count]
    partial_cost = [10 ** 9]
    partial = [None]
    nodes = [0]
    node_limit = 300000

    for axis_pos in axis_options:
        axis_cost = sum(abs(p - a["p"]) for p, a in zip(axis_pos, axes))
        if axis_cost > best_cost[0] or nodes[0] >= node_limit:
            break

        candidates = []
        current_masks = []
        fixed_mask = 0

        for pi, piece in enumerate(pieces):
            pos_masks = {}
            cells = piece["cells"]
            max_x = n - piece["w"]
            max_y = n - piece["h"]
            if max_x < 0 or max_y < 0:
                candidates.append([])
                current_masks.append(0)
                continue

            for ti, (tx, ty) in enumerate(targets):
                bit = 1 << ti
                for rx, ry in _orbit(tx, ty, axes, axis_pos, piece):
                    if not (0 <= rx < n and 0 <= ry < n):
                        continue
                    for dx, dy in cells:
                        px, py = rx - dx, ry - dy
                        if 0 <= px <= max_x and 0 <= py <= max_y:
                            if not piece["movable"] and (px, py) != current_positions[pi]:
                                continue
                            key = (px, py)
                            pos_masks[key] = pos_masks.get(key, 0) | bit

            cm = pos_masks.get(current_positions[pi], 0)
            current_masks.append(cm)
            if not piece["movable"]:
                fixed_mask |= cm
                candidates.append([])
                continue

            # For equal coverage, only the nearest translation is needed.
            by_mask = {}
            for pos, mask in pos_masks.items():
                dist = abs(pos[0] - piece["x"]) + abs(pos[1] - piece["y"])
                if axis_cost + dist > best_cost[0]:
                    continue
                old = by_mask.get(mask)
                if old is None or dist < old[0]:
                    by_mask[mask] = (dist, pos)
            entries = [(mask, dist, pos)
                       for mask, (dist, pos) in by_mask.items()]
            entries.sort(key=lambda e: e[1])
            candidates.append(entries)

        indices = []
        for entries in candidates:
            lists = [[] for _ in targets]
            for entry in entries:
                bits = entry[0]
                while bits:
                    low = bits & -bits
                    lists[low.bit_length() - 1].append(entry)
                    bits -= low
            indices.append(lists)

        chosen_positions = list(current_positions)
        movable = [i for i, p in enumerate(pieces) if p["movable"]]

        def search(unused, chosen_mask, cost):
            nodes[0] += 1
            if nodes[0] > node_limit or cost > best_cost[0]:
                return

            covered = fixed_mask | chosen_mask
            for i in unused:
                covered |= current_masks[i]

            count = covered.bit_count()
            if (count > partial_count[0] or
                    (count == partial_count[0] and count > initial_count
                     and cost < partial_cost[0])):
                partial_count[0] = count
                partial_cost[0] = cost
                partial[0] = (tuple(axis_pos), list(chosen_positions))

            missing = full & ~covered
            if not missing:
                if cost < best_cost[0]:
                    best_cost[0] = cost
                    best[0] = (tuple(axis_pos), list(chosen_positions))
                return
            if not unused:
                return

            # Branch on the uncovered target with the fewest candidate moves.
            target_index = -1
            smallest = 10 ** 9
            bits = missing
            while bits:
                low = bits & -bits
                ti = low.bit_length() - 1
                count_options = sum(len(indices[i][ti]) for i in unused)
                if count_options < smallest:
                    smallest = count_options
                    target_index = ti
                bits -= low
            if smallest == 0:
                return

            for i in unused:
                rest = [j for j in unused if j != i]
                for mask, dist, pos in indices[i][target_index]:
                    if cost + dist > best_cost[0]:
                        break
                    chosen_positions[i] = pos
                    search(rest, chosen_mask | mask, cost + dist)
                    chosen_positions[i] = current_positions[i]
                    if nodes[0] >= node_limit:
                        return

        search(movable, 0, axis_cost)
        if best_cost[0] == 0:
            break

    return best[0] if best[0] is not None else partial[0]


def _make_plan(state, solution):
    axis_dest, piece_dest = solution
    selected = state["selected"]

    axis_tasks = []
    for i, (axis, dest) in enumerate(zip(state["axes"], axis_dest)):
        if axis["movable"] and axis["p"] != dest:
            axis_tasks.append({"kind": "a", "i": i, "dest": dest})
    axis_tasks.sort(key=lambda t: 0 if selected == ("a", t["i"]) else 1)

    piece_tasks = []
    for i, (piece, dest) in enumerate(zip(state["pieces"], piece_dest)):
        if piece["movable"] and (piece["x"], piece["y"]) != tuple(dest):
            piece_tasks.append({"kind": "p", "i": i, "dest": tuple(dest)})
    piece_tasks.sort(key=lambda t: 0 if selected == ("p", t["i"]) else 1)

    return axis_tasks + piece_tasks


def _piece_cells(piece):
    return {(piece["x"] + dx, piece["y"] + dy)
            for dx, dy in piece["cells"]}


def _selected_from_frame(obs, state):
    scale, pad = state["scale"], state["pad"]
    marker = 2 if scale == 5 else 1
    n = state["n"]

    for i, axis in enumerate(state["axes"]):
        if not axis["movable"]:
            continue
        for k in range(n):
            x, y = (axis["p"], k) if axis["kind"] == "v" else (k, axis["p"])
            yy, xx = pad + y * scale, pad + x * scale
            if int(obs[yy, xx]) == 10 and int(obs[yy + marker, xx + marker]) == 0:
                return ("a", i)

    scores = []
    for i, piece in enumerate(state["pieces"]):
        score = 0
        for x, y in _piece_cells(piece):
            if 0 <= x < n and 0 <= y < n:
                yy, xx = pad + y * scale, pad + x * scale
                if int(obs[yy, xx]) == 5 and int(obs[yy + marker, xx + marker]) == 0:
                    score += 1
        if score:
            scores.append((score, i))
    if scores:
        return ("p", max(scores)[1])
    return None


def _click_for(state, kind, index):
    n = state["n"]
    scale, pad = state["scale"], state["pad"]
    pieces = state["pieces"]
    axes = state["axes"]

    if kind == "p":
        own = _piece_cells(pieces[index])
        others = set()
        for j, piece in enumerate(pieces):
            if j != index:
                others |= _piece_cells(piece)
        choices = sorted(own - others, key=lambda p: (p[1], p[0]))
        if choices:
            x, y = choices[0]
            return pad + y * scale + scale // 2, pad + x * scale + scale // 2
        return None

    axis = axes[index]
    occupied = set()
    for piece in pieces:
        occupied |= _piece_cells(piece)
    for k in range(n):
        x, y = (axis["p"], k) if axis["kind"] == "v" else (k, axis["p"])
        if (x, y) in occupied:
            continue
        crossing = False
        for j, other in enumerate(axes):
            if j == index or not other["movable"]:
                continue
            if ((other["kind"] == "v" and x == other["p"]) or
                    (other["kind"] == "h" and y == other["p"])):
                crossing = True
                break
        if not crossing:
            return pad + y * scale + scale // 2, pad + x * scale + scale // 2
    return None


def _original_matches(obs, state):
    scale, pad, n = state["scale"], state["pad"], state["n"]
    expected = set()
    for piece in state["pieces"]:
        expected |= _piece_cells(piece)
    observed = {(x, y) for y in range(n) for x in range(n)
                if int(obs[pad + y * scale, pad + x * scale]) == 5}
    return expected == observed


def act(obs, available, memory):
    # Verify after a batch of mirror moves: some withheld piece types can
    # rotate in response to mirror movement.
    if memory.get("verify"):
        memory["verify"] = False
        state = memory.get("state")
        if state is not None and not _original_matches(obs, state):
            memory.pop("state", None)
            memory.pop("plan", None)

    if "state" not in memory or not memory.get("plan"):
        state = _read(obs)
        solution = _solve(state)
        memory["state"] = state
        memory["plan"] = _make_plan(state, solution) if solution else []
        memory["waiting"] = False
        memory["cycle_count"] = 0

        if not memory["plan"]:
            # A no-op direction on a selected mirror also triggers the
            # simulator's goal check.
            sel = state["selected"]
            if sel and sel[0] == "a":
                axis = state["axes"][sel[1]]
                action = "ACTION1" if axis["kind"] == "v" else "ACTION3"
                return action, None, None
            if "RESET" in available and not memory.get("reset_used"):
                memory["reset_used"] = True
                memory.pop("state", None)
                return "RESET", None, None
            return "ACTION5", None, None

    state = memory["state"]
    plan = memory["plan"]

    if memory.get("waiting"):
        selected = _selected_from_frame(obs, state)
        if selected is not None:
            state["selected"] = selected
        memory["waiting"] = False

    while plan:
        task = plan[0]
        kind, i = task["kind"], task["i"]
        if kind == "a":
            axis = state["axes"][i]
            done = axis["p"] == task["dest"]
        else:
            piece = state["pieces"][i]
            done = (piece["x"], piece["y"]) == tuple(task["dest"])
        if not done:
            break
        plan.pop(0)
        if kind == "a" and (not plan or plan[0]["kind"] != "a"):
            memory["verify"] = True

    if not plan:
        memory.pop("state", None)
        memory.pop("plan", None)
        return act(obs, available, memory)

    task = plan[0]
    kind, i = task["kind"], task["i"]
    desired = (kind, i)

    if state["selected"] != desired:
        click = _click_for(state, kind, i)
        if click is not None and "ACTION6" in available:
            state["selected"] = desired
            return "ACTION6", int(click[0]), int(click[1])
        memory["waiting"] = True
        memory["cycle_count"] = memory.get("cycle_count", 0) + 1
        if memory["cycle_count"] > len(state["axes"]) + len(state["pieces"]) + 2:
            memory.pop("state", None)
            memory.pop("plan", None)
        return "ACTION5", None, None

    memory["cycle_count"] = 0
    if kind == "a":
        axis = state["axes"][i]
        dest = task["dest"]
        delta = 1 if dest > axis["p"] else -1
        axis["p"] += delta
        if axis["kind"] == "v":
            action = "ACTION4" if delta > 0 else "ACTION3"
        else:
            action = "ACTION2" if delta > 0 else "ACTION1"
    else:
        piece = state["pieces"][i]
        tx, ty = task["dest"]
        if piece["x"] != tx:
            delta = 1 if tx > piece["x"] else -1
            piece["x"] += delta
            action = "ACTION4" if delta > 0 else "ACTION3"
        else:
            delta = 1 if ty > piece["y"] else -1
            piece["y"] += delta
            action = "ACTION2" if delta > 0 else "ACTION1"
    return action, None, None

import numpy as np
import math
import random
import time
from collections import deque


def _components(a, colors):
    allowed = set(colors)
    h, w = a.shape
    seen = set()
    result = []
    for y in range(h):
        for x in range(w):
            if (x, y) in seen or int(a[y, x]) not in allowed:
                continue
            seen.add((x, y))
            queue = [(x, y)]
            cells = []
            for xx, yy in queue:
                cells.append((xx, yy))
                for nx, ny in ((xx-1, yy), (xx+1, yy),
                               (xx, yy-1), (xx, yy+1)):
                    if (0 <= nx < w and 0 <= ny < h
                            and (nx, ny) not in seen
                            and int(a[ny, nx]) in allowed):
                        seen.add((nx, ny))
                        queue.append((nx, ny))
            result.append(cells)
    return result


def _decode(obs):
    edges = [obs[0, :], obs[:, 0], obs[-1, :], obs[:, -1]]
    k = max(range(4), key=lambda i: int(np.count_nonzero(
        (edges[i] == 14) | (edges[i] == 0))))
    a = np.rot90(obs, -k).copy()

    lengths = []
    for color in (6, 8, 9, 11, 13, 15, 4):
        for y in range(1, 63):
            row = a[y] == color
            starts = np.flatnonzero(row & ~np.r_[False, row[:-1]])
            ends = np.flatnonzero(row & ~np.r_[row[1:], False])
            lengths.extend(int(e-s+1) for s, e in zip(starts, ends))
    scale = 0
    for length in lengths:
        scale = math.gcd(scale, length)
    if not 1 <= scale <= 4:
        scale = 1

    mask = a != 1
    mask[0, :] = False
    yy, xx = np.where(mask)
    pad = min(int(xx.min()), int(yy.min())) if len(xx) else 0
    n = (64-2*pad)//scale
    if not 8 <= n <= 64:
        n, pad, scale = 16, 0, 4

    coords = np.minimum(pad + np.arange(n)*scale + scale//2, 63)
    g = a[np.ix_(coords, coords)].astype(int)
    if scale == 1 and pad == 0:
        g[0, :] = 12
        for x in range(n):
            if g[1, x] == 6:
                g[0, x] = 4
    return g, k, scale, pad


def _piece(cells, kind, selected=False):
    x = min(p[0] for p in cells)
    y = min(p[1] for p in cells)
    return {
        "x": x,
        "y": y,
        "w": max(p[0] for p in cells)-x+1,
        "h": max(p[1] for p in cells)-y+1,
        "cells": tuple(sorted((xx-x, yy-y) for xx, yy in cells)),
        "kind": kind,
        "selected": selected
    }


def _set_valid(model, p):
    n = model["n"]
    fixed = model["fixed"]
    valid = set()
    for y in range(3, n-p["h"]+1):
        for x in range(n-p["w"]+1):
            good = True
            for cup in model["cups"]:
                if (x < cup["x"]+cup["w"]+1
                        and x+p["w"] > cup["x"]-1
                        and y < cup["y"]+cup["h"]+1
                        and y+p["h"] > cup["y"]-1):
                    good = False
                    break
            if good:
                for dx, dy in p["cells"]:
                    if fixed[(y+dy)*n+x+dx] != 0:
                        good = False
                        break
            if good:
                valid.add(y*n+x)
    valid.add(p["y"]*n+p["x"])
    p["valid"] = valid


def _build(g):
    n = len(g)
    cups = []
    targets = []
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))

    for cells in _components(g, (11, 13)):
        p = _piece(cells, "cup")
        cups.append(p)
        occupied = set(cells)
        target = None
        for yy in range(p["y"], p["y"]+p["h"]):
            for xx in range(p["x"], p["x"]+p["w"]):
                if (xx, yy) in occupied:
                    continue
                for d, (dx, dy) in enumerate(directions):
                    bx, by = xx+dx, yy+dy
                    px, py = (-1, 0) if dy else (0, -1)
                    if ((bx, by) in occupied
                            and (bx+px, by+py) in occupied
                            and (bx-px, by-py) in occupied
                            and (xx+px, yy+py) in occupied
                            and (xx-px, yy-py) in occupied):
                        target = (xx, yy, d)
        if target is None:
            target = (p["x"]+p["w"]//2, p["y"], 0)
        targets.append(target)

    objects = []
    for color in (8, 9, 15):
        for cells in _components(g, (color,)):
            p = _piece(cells, "corner" if color == 15 else "bar",
                       color == 9)
            if color == 9 and p["w"] > 1 and p["h"] > 1:
                p["kind"] = "corner"

            # Separate adjacent unselected straight pieces.
            if color == 8 and p["w"] > 1 and p["h"] > 1:
                horizontal = p["w"] >= p["h"]
                lines = {}
                for x, y in cells:
                    key, value = (y, x) if horizontal else (x, y)
                    lines.setdefault(key, []).append(value)
                for key, values in sorted(lines.items()):
                    values.sort()
                    run = [values[0]]
                    runs = []
                    for v in values[1:]:
                        if v == run[-1]+1:
                            run.append(v)
                        else:
                            runs.append(run)
                            run = [v]
                    runs.append(run)
                    for run in runs:
                        part = ([(v, key) for v in run] if horizontal
                                else [(key, v) for v in run])
                        objects.append(_piece(part, "bar"))
            else:
                objects.append(p)

    objects.sort(key=lambda p: (p["kind"] == "corner", p["y"], p["x"]))
    selected = next((i for i, p in enumerate(objects)
                     if p["selected"]), 0)

    fixed = [0]*(n*n)
    water = []
    for y in range(n):
        for x in range(n):
            c = int(g[y, x])
            if c in (1, 14):
                fixed[y*n+x] = -3
            elif c == 4:
                fixed[y*n+x] = -1
            elif c == 6:
                fixed[y*n+x] = -2
                water.append((x, y, 0))
    for i, cup in enumerate(cups):
        for dx, dy in cup["cells"]:
            fixed[(cup["y"]+dy)*n+cup["x"]+dx] = i+1

    model = {
        "n": n, "objects": objects, "cups": cups, "targets": targets,
        "fixed": fixed, "water": water, "selected": selected,
        "cache": {}, "tried": set(), "goal": None,
        "priority": None, "detour": None
    }
    for p in objects:
        _set_valid(model, p)
    return model


def _contains(p, x, y):
    return (p["x"] <= x < p["x"]+p["w"]
            and p["y"] <= y < p["y"]+p["h"])


def _synchronize(model, g, memory):
    components = _components(g, (9,))
    if not components or not model["objects"]:
        return
    observed = _piece(max(components, key=len), "bar", True)
    if observed["w"] > 1 and observed["h"] > 1:
        observed["kind"] = "corner"

    objects = model["objects"]
    matches = [
        i for i, p in enumerate(objects)
        if p["w"] == observed["w"] and p["h"] == observed["h"]
        and p["cells"] == observed["cells"]
    ]
    pending = memory.pop("click_target", None)
    chosen = None
    if pending is not None:
        intended, cx, cy = pending
        if intended < len(objects) and _contains(observed, cx, cy):
            if intended in matches:
                chosen = intended
            elif objects[intended]["kind"] == observed["kind"]:
                exact_others = [
                    i for i in matches
                    if objects[i]["x"] == observed["x"]
                    and objects[i]["y"] == observed["y"]
                ]
                chosen = exact_others[0] if exact_others else intended

    if chosen is None and matches:
        chosen = min(matches, key=lambda i:
                     abs(objects[i]["x"]-observed["x"]) +
                     abs(objects[i]["y"]-observed["y"]))
    if chosen is None:
        chosen = model["selected"]

    p = objects[chosen]
    shape_changed = (
        p["w"] != observed["w"] or p["h"] != observed["h"]
        or p["cells"] != observed["cells"]
    )
    p["x"], p["y"] = observed["x"], observed["y"]
    model["selected"] = chosen

    if shape_changed:
        p.update(observed)
        # Selection can reveal a larger piece whose visible fragments
        # were initially mistaken for separate objects.
        remove = set()
        for j, other in enumerate(objects):
            if j == chosen or other["kind"] != p["kind"]:
                continue
            if all((other["x"]+dx-p["x"], other["y"]+dy-p["y"])
                   in p["cells"] for dx, dy in other["cells"]):
                remove.add(j)
        if remove:
            new_selected = sum(j < chosen for j in range(len(objects))
                               if j not in remove)
            model["objects"] = [o for j, o in enumerate(objects)
                                if j not in remove]
            model["selected"] = new_selected
        for other in model["objects"]:
            _set_valid(model, other)
        model["cache"].clear()
        model["tried"].clear()
        model["goal"] = None
        model.pop("seed", None)
        model["priority"] = None
        model["detour"] = None


def _distances(piece, start, n):
    valid = piece["valid"]
    dist = {start: 0}
    q = deque([start])
    while q:
        pos = q.popleft()
        x, y = pos % n, pos // n
        neighbors = []
        if y:
            neighbors.append(pos-n)
        if y+1 < n:
            neighbors.append(pos+n)
        if x:
            neighbors.append(pos-1)
        if x+1 < n:
            neighbors.append(pos+1)
        for nxt in neighbors:
            if nxt in valid and nxt not in dist:
                dist[nxt] = dist[pos]+1
                q.append(nxt)
    return dist


def _simulate(model, layout):
    if layout in model["tried"]:
        return 10000.0, False
    cache = model["cache"]
    if layout in cache:
        return cache[layout]

    n = model["n"]
    objects = model["objects"]
    nc = len(model["cups"])
    grid = model["fixed"].copy()
    for i, (p, pos) in enumerate(zip(objects, layout)):
        x, y = pos % n, pos // n
        for dx, dy in p["cells"]:
            at = (y+dy)*n+x+dx
            if 0 <= at < len(grid) and grid[at] == 0:
                grid[at] = 100+i

    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    queue = list(model["water"])
    seen = {(y*n+x)*4+d for x, y, d in queue}
    filled = set()
    leaks = set()

    def cell(x, y):
        if 0 <= x < n and 0 <= y < n:
            return grid[y*n+x]
        return -1

    def add(x, y, d, force=False):
        if not (0 <= x < n and 0 <= y < n):
            return
        at = y*n+x
        if grid[at] == 0:
            grid[at] = -2
        elif not force:
            return
        key = at*4+d
        if key not in seen:
            seen.add(key)
            queue.append((x, y, d))

    head = 0
    while head < len(queue) and head < 4*n*n:
        x, y, d = queue[head]
        head += 1
        dx, dy = dirs[d]
        nx, ny = x+dx, y+dy
        hit = cell(nx, ny)
        if hit == 0:
            add(nx, ny, d)
            continue
        if hit == -2:
            at = (ny*n+nx)*4+d
            if at not in seen:
                seen.add(at)
                queue.append((nx, ny, d))
            continue
        if hit == -3:
            leaks.add((x, y, d))
            continue

        if dy:
            ax, ay, bx, by = x-1, y, x+1, y
        else:
            ax, ay, bx, by = x, y-1, x, y+1
        a, b = cell(ax, ay), cell(bx, by)

        if 1 <= hit <= nc:
            if a == hit and b == hit:
                filled.add(hit-1)
            else:
                if a == 0:
                    add(ax, ay, d)
                if b == 0:
                    add(bx, by, d)
        elif hit >= 100:
            p = objects[hit-100]
            if p["kind"] == "bar":
                if a == 0:
                    add(ax, ay, d)
                if b == 0:
                    add(bx, by, d)
            else:
                if a == hit and b == 0:
                    nd = (d+1) % 4
                    ux, uy = dirs[nd]
                    add(x+ux, y+uy, nd, True)
                if b == hit and a == 0:
                    nd = (d-1) % 4
                    ux, uy = dirs[nd]
                    add(x+ux, y+uy, nd, True)
                else:
                    if cell(ax, ay) == 0:
                        add(ax, ay, d)
                    if cell(bx, by) == 0:
                        add(bx, by, d)

    success = len(filled) == nc and not leaks
    if success:
        result = (0.0, True)
    else:
        score = 60.0*(nc-len(filled)) + 12.0*len(leaks)
        for i, (tx, ty, td) in enumerate(model["targets"]):
            if i in filled:
                continue
            best = 2*n
            for x, y, d in queue:
                distance = abs(x-tx)+abs(y-ty)
                if d != td:
                    distance += max(4, n//2)
                if distance < best:
                    best = distance
            score += 3.0*best
        for x, y, d in leaks:
            if model["targets"]:
                score += 0.75*min(
                    abs(x-tx)+abs(y-ty)
                    for tx, ty, td in model["targets"])
        if head >= 4*n*n:
            score += 100
        result = (score, False)

    if len(cache) > 45000:
        cache.clear()
    cache[layout] = result
    return result


def _search(model, rng, deadline):
    n = model["n"]
    pieces = model["objects"]
    current = tuple(p["y"]*n+p["x"] for p in pieces)
    selected = model["selected"]
    distances = [_distances(p, pos, n)
                 for p, pos in zip(pieces, current)]
    choices = [list(d) for d in distances]

    def cost(layout):
        total = 0
        for i, (pos, old) in enumerate(zip(layout, current)):
            total += distances[i].get(pos, 1000)
            if pos != old and i != selected:
                total += 1
        return total

    def value(layout):
        score, success = _simulate(model, layout)
        return score + 0.045*cost(layout), success

    best = current
    best_value, successful = value(best)
    seed = model.get("seed")
    if (seed and len(seed) == len(pieces)
            and all(seed[i] in distances[i] for i in range(len(pieces)))):
        v, ok = value(seed)
        if v < best_value:
            best, best_value, successful = seed, v, ok

    pool = [(best_value, best)]
    state = best
    state_value = best_value
    iteration = 0
    phase = 0
    count = len(pieces)

    while count and not successful and time.monotonic() < deadline:
        if phase == 0:
            if rng.random() < 0.22:
                state = tuple(rng.choice(c) for c in choices)
            else:
                state = rng.choice(pool)[1]
            state_value, _ = value(state)

        proposal = list(state)
        mutations = 2 if count > 1 and rng.random() < 0.12 else 1
        for _ in range(mutations):
            i = rng.randrange(count)
            old = proposal[i]
            x, y = old % n, old // n
            r = rng.random()
            if r < 0.40:
                if rng.random() < 0.5:
                    x += rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
                else:
                    y += rng.choice((-3, -2, -1, 1, 2, 3))
                pos = y*n+x
                if 0 <= x < n and 0 <= y < n and pos in distances[i]:
                    proposal[i] = pos
            elif r < 0.78:
                if rng.random() < 0.65:
                    x = rng.randrange(n)
                else:
                    y = rng.randrange(3, n)
                pos = y*n+x
                if pos in distances[i]:
                    proposal[i] = pos
            else:
                proposal[i] = rng.choice(choices[i])

        proposal = tuple(proposal)
        v, ok = value(proposal)
        temperature = 55.0 * (0.008 ** (phase/420.0))
        if (v <= state_value or rng.random() < math.exp(
                min(0.0, (state_value-v)/max(0.2, temperature)))):
            state, state_value = proposal, v

        if v < best_value:
            best, best_value = proposal, v
        if ok:
            best, best_value, successful = proposal, v, True
            break

        if iteration % 19 == 0:
            if all(proposal != item[1] for item in pool):
                pool.append((v, proposal))
                pool.sort(key=lambda item: item[0])
                pool = pool[:14]
        iteration += 1
        phase = (phase+1) % 421

    if successful:
        best_cost = cost(best)
        for _ in range(2):
            order = sorted(range(count),
                           key=lambda i: distances[i].get(best[i], 1000),
                           reverse=True)
            for i in order:
                old_cost = distances[i].get(best[i], 1000)
                if best[i] != current[i] and i != selected:
                    old_cost += 1
                ordered = sorted(choices[i], key=lambda pos:
                                 distances[i][pos] +
                                 int(pos != current[i] and i != selected))
                for pos in ordered:
                    if time.monotonic() >= deadline:
                        break
                    c = distances[i][pos] + int(
                        pos != current[i] and i != selected)
                    if c >= old_cost:
                        break
                    candidate = list(best)
                    candidate[i] = pos
                    candidate = tuple(candidate)
                    if _simulate(model, candidate)[1]:
                        candidate_cost = cost(candidate)
                        if candidate_cost < best_cost:
                            best, best_cost = candidate, candidate_cost
                            break
                if time.monotonic() >= deadline:
                    break
            if time.monotonic() >= deadline:
                break

    model["seed"] = best
    if successful:
        model["goal"] = best
    return best


def _bbox(p, pos=None, n=None):
    if pos is None:
        x, y = p["x"], p["y"]
    else:
        x, y = pos % n, pos // n
    return x, y, x+p["w"], y+p["h"]


def _intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _click_point(model, index, overrides=None):
    objects = model["objects"]
    n = model["n"]
    overrides = overrides or {}
    p = objects[index]
    box = _bbox(p, overrides.get(index), n)
    x, y = box[0], box[1]

    # The game's click routine tests every bar before every corner,
    # and uses whole bounding boxes rather than opaque pixels.
    blockers = []
    for j, other in enumerate(objects):
        if j == index:
            continue
        if p["kind"] == "bar" and other["kind"] != "bar":
            continue
        blockers.append(_bbox(other, overrides.get(j), n))

    candidates = [(x+dx, y+dy) for dx, dy in p["cells"]]
    occupied_candidates = set(candidates)
    candidates.extend(
        (xx, yy)
        for yy in range(y, y+p["h"])
        for xx in range(x, x+p["w"])
        if (xx, yy) not in occupied_candidates
    )
    for xx, yy in candidates:
        if not any(b[0] <= xx < b[2] and b[1] <= yy < b[3]
                   for b in blockers):
            return xx, yy
    return None


def _plan_detour(model, target, goal):
    objects = model["objects"]
    n = model["n"]
    target_box = _bbox(objects[target])
    selected = model["selected"]
    possibilities = []

    for j, p in enumerate(objects):
        if j == target:
            continue
        if objects[target]["kind"] == "bar" and p["kind"] != "bar":
            continue
        if not _intersects(_bbox(p), target_box):
            continue
        point = _click_point(model, j)
        if j != selected and point is None:
            continue
        start = p["y"]*n+p["x"]
        distances = _distances(p, start, n)
        for pos, distance in distances.items():
            if pos == start:
                continue
            box = _bbox(p, pos, n)
            if _intersects(box, target_box):
                continue
            overlaps = sum(
                _intersects(box, _bbox(other))
                for i, other in enumerate(objects) if i != j)
            score = distance + int(j != selected) + 1.5*overlaps
            if pos == goal[j]:
                score -= 1.5
            possibilities.append((score, j, pos))

    if not possibilities:
        return False
    _, j, pos = min(possibilities)
    model["priority"] = target
    model["detour"] = (j, pos)
    return True


def _display_point(x, y, k):
    if k == 1:
        return y, 63-x
    if k == 2:
        return 63-x, 63-y
    if k == 3:
        return 63-y, x
    return x, y


def _select(model, index, point, k, scale, pad, memory):
    cx, cy = point
    x = pad + scale*cx + scale//2
    y = pad + scale*cy + scale//2
    sx, sy = _display_point(x, y, k)
    memory["last_action"] = "ACTION6"
    memory["click_target"] = (index, cx, cy)
    return "ACTION6", int(sy), int(sx)


def _reset(memory):
    memory.pop("model", None)
    memory.pop("click_target", None)
    memory["last_action"] = "RESET"
    return "RESET", None, None


def act(obs, available, memory):
    started = time.monotonic()
    g, k, scale, pad = _decode(np.asarray(obs))
    model = memory.get("model")

    rng = random.Random()
    if model is None or model["n"] != len(g):
        model = _build(g)
        memory["model"] = model
        memory.pop("click_target", None)
        rng.seed(173 + int(np.sum(g))*19)
    else:
        if "rng" in memory:
            rng.setstate(memory["rng"])
        else:
            rng.seed(173)
        _synchronize(model, g, memory)

        if memory.get("last_action") == "ACTION5":
            previous = memory.get("last_layout")
            if previous is not None and len(previous) == len(model["objects"]):
                model["tried"].add(previous)
                model["cache"].pop(previous, None)
            model["goal"] = None
            model.pop("seed", None)
            model["priority"] = None
            model["detour"] = None

    pieces = model["objects"]
    n = model["n"]
    if not pieces:
        return ("ACTION5" if "ACTION5" in available else available[0],
                None, None)

    goal = model["goal"]
    if goal is None:
        goal = _search(model, rng, started+8.5)
    memory["rng"] = rng.getstate()

    current = tuple(p["y"]*n+p["x"] for p in pieces)
    if current == goal and _simulate(model, goal)[1]:
        memory["last_action"] = "ACTION5"
        memory["last_layout"] = current
        return "ACTION5", None, None

    unfinished = [i for i in range(len(pieces)) if current[i] != goal[i]]
    selected = model["selected"]

    detour = model.get("detour")
    if detour is not None and current[detour[0]] == detour[1]:
        model["detour"] = None
        detour = None
    priority = model.get("priority")
    if priority is not None and priority not in unfinished:
        model["priority"] = None
        priority = None

    if detour is not None:
        desired, destination = detour
    elif priority is not None:
        desired, destination = priority, goal[priority]
    elif selected in unfinished:
        desired, destination = selected, goal[selected]
    elif unfinished:
        desired = min(
            unfinished,
            key=lambda i: (
                _click_point(model, i) is None,
                abs(current[i] % n-goal[i] % n) +
                abs(current[i] // n-goal[i] // n)))
        destination = goal[desired]
    else:
        point = _click_point(model, selected)
        if point is None:
            p = pieces[selected]
            point = (p["x"], p["y"])
        return _select(model, selected, point, k, scale, pad, memory)

    if desired != selected:
        point = _click_point(model, desired)
        if point is None:
            if not _plan_detour(model, desired, goal):
                return _reset(memory)
            desired, destination = model["detour"]
            if desired != selected:
                point = _click_point(model, desired)
                if point is None:
                    return _reset(memory)
                return _select(model, desired, point, k, scale, pad, memory)
        else:
            return _select(model, desired, point, k, scale, pad, memory)

    p = pieces[desired]
    reverse = _distances(p, destination, n)
    pos = current[desired]
    x, y = p["x"], p["y"]
    moves = [
        (pos-n, 1, 0, -1),
        (pos+n, 2, 0, 1),
        (pos-1, 3, -1, 0),
        (pos+1, 4, 1, 0)
    ]
    legal = [
        item for item in moves
        if 0 <= x+item[2] < n and 0 <= y+item[3] < n
        and item[0] in reverse
    ]
    if not legal:
        return _reset(memory)

    nxt, canonical_action, dx, dy = min(
        legal, key=lambda item: reverse[item[0]])
    p["x"] += dx
    p["y"] += dy

    inverse_actions = {
        0: {1: 1, 2: 2, 3: 3, 4: 4},
        1: {1: 3, 2: 4, 3: 2, 4: 1},
        2: {1: 2, 2: 1, 3: 4, 4: 3},
        3: {1: 4, 2: 3, 3: 1, 4: 2}
    }
    action = "ACTION" + str(inverse_actions[k][canonical_action])
    memory["last_action"] = action
    return action, None, None

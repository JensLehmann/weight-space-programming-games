import math
import random
import time
from collections import deque

import numpy as np


def _components(g, colours):
    n = len(g)
    wanted = set(colours)
    seen = set()
    result = []
    for y in range(n):
        for x in range(n):
            if (x, y) in seen or int(g[y, x]) not in wanted:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            component = []
            while stack:
                px, py = stack.pop()
                component.append((px, py))
                for qx, qy in (
                    (px - 1, py), (px + 1, py),
                    (px, py - 1), (px, py + 1),
                ):
                    if (0 <= qx < n and 0 <= qy < n
                            and (qx, qy) not in seen
                            and int(g[qy, qx]) in wanted):
                        seen.add((qx, qy))
                        stack.append((qx, qy))
            result.append(component)
    return result


def _rotation(obs):
    edges = [obs[0, :], obs[:, 0], obs[-1, :], obs[:, -1]]
    scores = [int(np.count_nonzero((e == 14) | (e == 0))) for e in edges]
    return max(range(4), key=lambda k: scores[k])


def _decode(obs):
    k = _rotation(obs)
    a = np.rot90(obs, -k)

    # The widths of sprite-colour runs reveal the integer camera scale.
    scale = 0
    informative = {6, 8, 9, 11, 13, 15}
    for row in a[1:]:
        start = 0
        while start < 64:
            end = start + 1
            while end < 64 and row[end] == row[start]:
                end += 1
            if int(row[start]) in informative:
                scale = math.gcd(scale, end - start)
            start = end
    if not scale:
        scale = 4
    scale = max(1, min(8, scale))

    best = None
    lo = max(6, 64 // (scale + 1) + 1)
    hi = min(64, 64 // scale)
    for n in range(lo, hi + 1):
        if 64 // n != scale:
            continue
        off = (64 - n * scale) // 2
        outside = np.ones((64, 64), dtype=bool)
        outside[off:off + n * scale, off:off + n * scale] = False
        outside[0, :] = False
        penalty = 3 * int(np.count_nonzero(outside & (a != 1)))

        g = np.full((n, n), 12, dtype=np.int8)
        mismatch = 0
        for y in range(n):
            y0 = off + y * scale
            for x in range(n):
                x0 = off + x * scale
                block = a[y0:y0 + scale, x0:x0 + scale]
                if y0 == 0:
                    block = block[1:]
                values = block.reshape(-1)
                if not len(values):
                    continue
                counts = np.bincount(values, minlength=16)
                colour = int(np.argmax(counts))
                g[y, x] = colour
                mismatch += len(values) - int(counts[colour])

        # Sources normally occupy the first two logical rows. This also
        # distinguishes a viewport from a larger, padding-containing one.
        sy, sx = np.where(g == 4)
        if len(sy):
            penalty += 20 * int(np.min(sy))
        wy, wx = np.where(g == 6)
        if len(wy):
            penalty += 20 * abs(int(np.min(wy)) - 1)
        score = penalty + mismatch + n * 0.001
        if best is None or score < best[0]:
            best = (score, n, off, g)

    if best is None:
        n = 16
        scale = 4
        off = 0
        g = a[2::4, 2::4].copy()
    else:
        _, n, off, g = best
    return k, n, scale, off, g


def _make_world(obs):
    k, n, scale, off, g = _decode(obs)
    stride = n + 2
    size = stride * stride

    def position(x, y):
        return (y + 1) * stride + x + 1

    buckets = []
    for cells in _components(g, {11, 13}):
        xs = [p[0] for p in cells]
        ys = [p[1] for p in cells]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        occupied = set(cells)
        holes = [
            (x, y)
            for y in range(y0, y1 + 1)
            for x in range(x0, x1 + 1)
            if (x, y) not in occupied
        ]
        candidates = []
        for x, y in holes:
            if y == y0:
                candidates.append((x, y, 0))
            if x == x0:
                candidates.append((x, y, 1))
            if y == y1:
                candidates.append((x, y, 2))
            if x == x1:
                candidates.append((x, y, 3))
        if candidates:
            tx, ty, direction = min(
                candidates,
                key=lambda q: abs(2 * q[0] - x0 - x1)
                + abs(2 * q[1] - y0 - y1),
            )
        else:
            tx, ty, direction = (x0 + x1) // 2, y0, 0
        buckets.append({
            "cells": cells,
            "box": (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
            "mouth": (tx, ty, direction),
        })

    pieces = []
    for colour in (8, 9, 15):
        for cells in _components(g, {colour}):
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            w, h = x1 - x0 + 1, y1 - y0 + 1
            full = len(cells) == w * h
            is_elbow = colour == 15 or (
                colour == 9 and not full and w > 1 and h > 1
            )

            # Separate touching parallel bars when the component is rectangular.
            groups = [cells]
            if not is_elbow and w > 1 and h > 1:
                if w >= h:
                    groups = [
                        [(x, y) for x, yy in cells if yy == y]
                        for y in range(y0, y1 + 1)
                    ]
                else:
                    groups = [
                        [(x, y) for xx, y in cells if xx == x]
                        for x in range(x0, x1 + 1)
                    ]
            for group in groups:
                if not group:
                    continue
                px = min(x for x, y in group)
                py = min(y for x, y in group)
                pw = max(x for x, y in group) - px + 1
                ph = max(y for x, y in group) - py + 1
                mask = [(x - px, y - py) for x, y in group]
                pieces.append({
                    "kind": "elbow" if is_elbow else "bar",
                    "w": pw,
                    "h": ph,
                    "mask": mask,
                    "offsets": [dy * stride + dx for dx, dy in mask],
                    "pos": position(px, py),
                    "selected": colour == 9,
                })

    # Bars precede elbows in the simulator's relevant sprite ordering.
    pieces.sort(key=lambda p: p["kind"] == "elbow")
    selected = next(
        (i for i, p in enumerate(pieces) if p["selected"]), None
    )

    base = [1] * size
    seeds = []
    movable_cells = set()
    for p in pieces:
        x = p["pos"] % stride - 1
        y = p["pos"] // stride - 1
        for dx, dy in p["mask"]:
            movable_cells.add((x + dx, y + dy))

    for y in range(n):
        for x in range(n):
            colour = int(g[y, x])
            q = position(x, y)
            if (x, y) in movable_cells:
                base[q] = 0
            elif colour == 12:
                base[q] = 0
            elif colour in (1, 14):
                base[q] = 2
            elif colour == 6:
                base[q] = 3
                seeds.append(q)
            elif colour in (11, 13):
                base[q] = 0
            else:
                base[q] = 1

    for i, b in enumerate(buckets):
        for x, y in b["cells"]:
            base[position(x, y)] = 5 + i

    # A source can create a fresh starting liquid pixel at spill time.
    for y in range(n - 1):
        for x in range(n):
            if int(g[y, x]) == 4 and (x, y) not in movable_cells:
                q = position(x, y + 1)
                if base[q] == 0:
                    base[q] = 3
                    seeds.append(q)
    seeds = list(dict.fromkeys(seeds))

    elbow_count = 0
    for p in pieces:
        if p["kind"] == "bar":
            p["code"] = 4
        else:
            p["code"] = 5 + len(buckets) + elbow_count
            elbow_count += 1

        domain = []
        for y in range(3, n - p["h"] + 1):
            for x in range(n - p["w"] + 1):
                legal = True
                for b in buckets:
                    rx, ry, rw, rh = b["box"]
                    if (x < rx + rw + 1 and x + p["w"] > rx - 1
                            and y < ry + rh + 1 and y + p["h"] > ry - 1):
                        legal = False
                        break
                if not legal:
                    continue
                q = position(x, y)
                if all(base[q + d] == 0 for d in p["offsets"]):
                    domain.append(q)
        p["domain"] = domain
        p["allowed"] = set(domain)

    # Precomputed geometric guidance for unsuccessful simulations.
    proximity = []
    for b in buckets:
        tx, ty, td = b["mouth"]
        table = [0.0] * (size * 4)
        for q in range(size):
            x, y = q % stride - 1, q // stride - 1
            distance = abs(x - tx) + abs(y - ty)
            for d in range(4):
                table[q * 4 + d] = distance + (0 if d == td else 6)
        proximity.append(table)

    terminal_cost = [0.0] * (size * 4)
    for q in range(size):
        x, y = q % stride - 1, q // stride - 1
        for d in range(4):
            values = []
            for b in buckets:
                tx, ty, td = b["mouth"]
                if d in (0, 2):
                    distance = abs(x - tx) + 0.35 * abs(y - ty)
                else:
                    distance = abs(y - ty) + 0.35 * abs(x - tx)
                values.append(distance + (0 if d == td else 7))
            terminal_cost[q * 4 + d] = min(values) if values else n

    return {
        "k": k, "n": n, "scale": scale, "off": off,
        "stride": stride, "size": size,
        "base": base, "seeds": seeds, "buckets": buckets,
        "pieces": pieces,
        "positions": [p["pos"] for p in pieces],
        "selected": selected,
        "proximity": proximity,
        "terminal_cost": terminal_cost,
        "cache": {},
    }


def _simulate(world, positions):
    cache = world["cache"]
    key = tuple(positions)
    cached = cache.get(key)
    if cached is not None:
        return cached

    board = world["base"].copy()
    overlap = 0
    for piece, q in zip(world["pieces"], positions):
        code = piece["code"]
        for offset in piece["offsets"]:
            cell = q + offset
            if board[cell] == 0:
                board[cell] = code
            else:
                overlap += 1

    stride = world["stride"]
    directions = (stride, 1, -stride, -1)
    side_offsets = ((-1, 1), (-stride, stride),
                    (-1, 1), (-stride, stride))
    bucket_count = len(world["buckets"])
    elbow_start = 5 + bucket_count

    queue = [q * 4 for q in world["seeds"]]
    head = 0
    seen = [0] * world["size"]
    visited_states = []
    leaks = []
    hitmask = 0

    while head < len(queue):
        state = queue[head]
        head += 1
        q, direction = state >> 2, state & 3
        bit = 1 << direction
        if seen[q] & bit:
            continue
        seen[q] |= bit
        visited_states.append(state)

        forward = q + directions[direction]
        code = board[forward]

        if code == 0:
            board[forward] = 3
            queue.append(forward * 4 + direction)
            continue
        if code == 3:
            queue.append(forward * 4 + direction)
            continue
        if code == 2:
            leaks.append(state)
            continue
        if code == 1:
            continue

        negative_offset, positive_offset = side_offsets[direction]
        negative = q + negative_offset
        positive = q + positive_offset
        nc, pc = board[negative], board[positive]

        if 5 <= code < elbow_start:
            if nc == code and pc == code:
                hitmask |= 1 << (code - 5)
                continue

        elif code >= elbow_start:
            # The simulator has two separate turn tests, with the fallback
            # attached to the second one.
            if nc == code and pc == 0:
                turned = (direction + 1) & 3
                nq = q + directions[turned]
                board[nq] = 3
                queue.append(nq * 4 + turned)

            if pc == code and nc == 0:
                turned = (direction - 1) & 3
                nq = q + directions[turned]
                board[nq] = 3
                queue.append(nq * 4 + turned)
                continue

        # Bars, unsuccessful receiver approaches, and corner fallbacks split.
        if board[negative] == 0:
            board[negative] = 3
            queue.append(negative * 4 + direction)
        if board[positive] == 0:
            board[positive] = 3
            queue.append(positive * 4 + direction)

    missing = [
        i for i in range(bucket_count) if not (hitmask & (1 << i))
    ]
    success = not missing and not leaks

    if success:
        score = 0.0
    else:
        score = 40.0 * len(missing) + 22.0 * len(leaks)
        for i in missing:
            table = world["proximity"][i]
            nearest = min(
                (table[state] for state in visited_states),
                default=2 * world["n"],
            )
            score += 2.0 * nearest
        costs = world["terminal_cost"]
        score += 1.5 * sum(costs[state] for state in leaks)
        score += 1.5 * overlap

    result = (score, success)
    if len(cache) < 150000:
        cache[key] = result
    return result


def _distances(world, piece, start):
    allowed = piece["allowed"]
    distances = {start: 0}
    queue = deque([start])
    stride = world["stride"]
    while queue:
        q = queue.popleft()
        nd = distances[q] + 1
        for nq in (q - stride, q + stride, q - 1, q + 1):
            if nq in allowed and nq not in distances:
                distances[nq] = nd
                queue.append(nq)
    return distances


def _search(world, deadline, seed):
    pieces = world["pieces"]
    starts = tuple(world["positions"])
    maps = [
        _distances(world, p, q)
        for p, q in zip(pieces, starts)
    ]
    domains = [
        [q for q in p["domain"] if q in distance]
        for p, distance in zip(pieces, maps)
    ]
    for i in range(len(domains)):
        if starts[i] not in domains[i]:
            domains[i].append(starts[i])

    selected = world["selected"]
    rng = random.Random(seed)
    count = len(pieces)

    def movement_cost(positions):
        total = 0
        for i, q in enumerate(positions):
            total += maps[i].get(q, 10000)
            if q != starts[i] and i != selected:
                total += 1
        return total

    def fitness(positions):
        raw, success = _simulate(world, positions)
        return raw + 0.075 * movement_cost(positions), success

    best = starts
    best_score, best_success = fitness(best)
    rounds = 0
    checks = 0

    while time.perf_counter() < deadline:
        if rounds == 0:
            current = list(starts)
        else:
            current = list(best)
            if rng.random() < 0.30:
                indices = list(range(count))
            else:
                amount = 1 if count == 1 else rng.choice((1, 2, 2, count))
                indices = rng.sample(range(count), min(amount, count))
            for i in indices:
                if domains[i]:
                    current[i] = rng.choice(domains[i])

        current_score, current_success = fitness(current)
        if current_score < best_score:
            best, best_score, best_success = (
                tuple(current), current_score, current_success
            )

        for sweep in range(4):
            order = list(range(count))
            rng.shuffle(order)
            improved = False

            for i in order:
                old = current[i]
                local_position = old
                local_score = current_score
                local_success = current_success

                candidates = domains[i]
                # The original/current position is naturally favored by cost.
                for q in candidates:
                    current[i] = q
                    value, success = fitness(current)
                    checks += 1
                    if value < best_score - 1e-9:
                        best = tuple(current)
                        best_score = value
                        best_success = success
                    if value < local_score - 1e-9:
                        local_position = q
                        local_score = value
                        local_success = success
                    if (checks & 63) == 0:
                        if time.perf_counter() >= deadline:
                            return best, best_success, maps

                current[i] = local_position
                if local_position != old:
                    improved = True
                current_score = local_score
                current_success = local_success

            if not improved:
                break
        rounds += 1

    return best, best_success, maps


def _plan(world, targets, maps, success):
    starts = world["positions"]
    selected = world["selected"]
    stride = world["stride"]
    changed = [i for i in range(len(starts)) if starts[i] != targets[i]]

    changed.sort(key=lambda i: (
        i != selected,
        maps[i].get(targets[i], 10000),
    ))

    plan = []
    for i in changed:
        if selected != i:
            plan.append((6, i, starts[i]))
            selected = i

        target = targets[i]
        distances = maps[i]
        if target not in distances:
            continue
        route = [target]
        q = target
        while q != starts[i]:
            predecessor = None
            for nq in (q - stride, q + stride, q - 1, q + 1):
                if distances.get(nq) == distances[q] - 1:
                    predecessor = nq
                    break
            if predecessor is None:
                route = []
                break
            q = predecessor
            route.append(q)
        route.reverse()

        for before, after in zip(route, route[1:]):
            delta = after - before
            internal = {
                -stride: 1, stride: 2, -1: 3, 1: 4
            }[delta]
            plan.append((internal, i, after))

    if success:
        plan.append((5, None, None))
    return plan


def _selection_matches(obs, world):
    selected = world["selected"]
    if selected is None:
        return True
    a = np.rot90(obs, -world["k"])
    ys, xs = np.where(a == 9)
    if not len(xs):
        return False

    piece = world["pieces"][selected]
    q = world["positions"][selected]
    stride = world["stride"]
    x, y = q % stride - 1, q // stride - 1
    scale, off = world["scale"], world["off"]
    return (
        int(xs.min()) == off + x * scale
        and int(ys.min()) == off + y * scale
        and int(xs.max()) == off + (x + piece["w"]) * scale - 1
        and int(ys.max()) == off + (y + piece["h"]) * scale - 1
    )


def _click_coordinates(world, index):
    stride = world["stride"]
    piece = world["pieces"][index]
    q = world["positions"][index]
    x, y = q % stride - 1, q // stride - 1

    candidates = sorted(
        piece["mask"],
        key=lambda p: abs(2 * p[0] - piece["w"] + 1)
        + abs(2 * p[1] - piece["h"] + 1),
    )
    chosen = candidates[0]
    for dx, dy in candidates:
        px, py = x + dx, y + dy
        clear = True
        for j, other in enumerate(world["pieces"]):
            if j == index:
                continue
            oq = world["positions"][j]
            ox, oy = oq % stride - 1, oq // stride - 1
            if ox <= px < ox + other["w"] and oy <= py < oy + other["h"]:
                clear = False
                break
        if clear:
            chosen = (dx, dy)
            break

    dx, dy = chosen
    scale, off = world["scale"], world["off"]
    raw_x = off + (x + dx) * scale + scale // 2
    raw_y = off + (y + dy) * scale + scale // 2
    k = world["k"]
    if k == 1:
        display_x, display_y = raw_y, 63 - raw_x
    elif k == 2:
        display_x, display_y = 63 - raw_x, 63 - raw_y
    elif k == 3:
        display_x, display_y = 63 - raw_y, raw_x
    else:
        display_x, display_y = raw_x, raw_y
    return int(display_y), int(display_x)


def act(obs, available, memory):
    started = time.perf_counter()
    memory["calls"] = memory.get("calls", 0) + 1

    world = memory.get("world")
    if world is not None and not _selection_matches(obs, world):
        # Unexpected motion or selection: reconstruct instead of blindly
        # continuing the old plan.
        memory.pop("plan", None)
        world = None

    if world is None:
        world = _make_world(obs)
        memory["world"] = world
        memory["plan"] = []

    if not world["pieces"] or not world["buckets"]:
        action = "ACTION5" if "ACTION5" in available else available[0]
        return action, None, None

    plan = memory.get("plan", [])
    if not plan:
        seed = (
            7919 * memory["calls"]
            + 131 * world["n"]
            + sum((i + 1) * q for i, q in enumerate(world["positions"]))
        )
        targets, success, maps = _search(
            world, started + 8.0, seed
        )
        plan = _plan(world, targets, maps, success)

        if not success:
            # Spend further computation on the next call rather than committing
            # to a spill that the model already predicts will fail.
            plan = plan[:1]
        if not plan:
            index = memory["calls"] % len(world["pieces"])
            plan = [(6, index, world["positions"][index])]
        memory["plan"] = plan

    internal, index, destination = memory["plan"].pop(0)

    if internal == 6:
        row, col = _click_coordinates(world, index)
        world["selected"] = index
        action = "ACTION6"
    elif internal == 5:
        row = col = None
        action = "ACTION5"
        world["selected"] = None
        # A failed spill is recoverable; if the episode continues, re-read it.
        memory.pop("world", None)
        memory.pop("plan", None)
    else:
        external_for_internal = (
            (0, 1, 2, 3, 4),
            (0, 3, 4, 2, 1),
            (0, 2, 1, 4, 3),
            (0, 4, 3, 1, 2),
        )
        external = external_for_internal[world["k"]][internal]
        action = "ACTION" + str(external)
        row = col = None
        world["positions"][index] = destination
        world["selected"] = index

    if action not in available:
        memory.pop("world", None)
        memory.pop("plan", None)
        fallback = "ACTION5" if "ACTION5" in available else available[0]
        return fallback, None, None
    return action, row, col

import numpy as np
from collections import deque
from itertools import product


def _board(obs):
    """Recover the camera grid, cell colors, and interface markers."""
    a = np.asarray(obs)
    region = a[:, :63]
    ys, xs = np.where(region != 5)
    if len(xs) == 0:
        return None

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    candidates = []

    for scale in range(2, 7):
        dims_x = []
        dims_y = []
        for n in range(1, 64 // scale + 1):
            size = n * scale
            pad = (64 - size) // 2
            if pad <= xmin and pad + size - 1 >= xmax:
                if xmin - pad <= 2 and pad + size - 1 - xmax <= 2:
                    dims_x.append((n, pad))
            if pad <= ymin and pad + size - 1 >= ymax:
                if ymin - pad <= 2 and pad + size - 1 - ymax <= 2:
                    dims_y.append((n, pad))

        for w, ox in dims_x:
            for h, oy in dims_y:
                if min(64 // w, 64 // h) != scale:
                    continue
                base = np.empty((h, w), dtype=np.int16)
                marker = np.empty((h, w), dtype=np.int16)
                mixed = 0

                for y in range(h):
                    for x in range(w):
                        block = a[
                            oy + y * scale:oy + (y + 1) * scale,
                            ox + x * scale:ox + (x + 1) * scale,
                        ]
                        # The energy display occupies display column 63.
                        if ox + (x + 1) * scale == 64:
                            block = block[:, :-1]
                        counts = np.bincount(block.ravel().astype(int), minlength=16)
                        color = int(np.argmax(counts))
                        base[y, x] = color
                        mixed += int(block.size - counts[color])

                        dx = ((64 - w * scale + 1) // 2) - ox
                        dy = ((64 - h * scale + 1) // 2) - oy
                        if scale == 3:
                            mx, my = dx, dy
                        elif scale == 4:
                            mx, my = dx + 1, dy + 1
                        elif scale == 5:
                            mx, my = dx + 2, dy + 2
                        else:
                            # The provided renderer only specifies markers
                            # for scales 3, 4 and 5.
                            mx = my = scale // 2
                        mx = min(scale - 1, max(0, mx))
                        my = min(scale - 1, max(0, my))
                        marker[y, x] = int(a[oy + y * scale + my,
                                                ox + x * scale + mx])

                edge_error = (
                    xmin - ox + ox + w * scale - 1 - xmax
                    + ymin - oy + oy + h * scale - 1 - ymax
                )
                score = mixed / max(1, w * h * scale * scale)
                score += edge_error * 0.004
                candidates.append((score, -scale, w, h, ox, oy,
                                   scale, base, marker))

    if not candidates:
        return None
    candidates.sort(key=lambda z: (z[0], z[1]))
    _, _, w, h, ox, oy, scale, base, marker = candidates[0]
    return w, h, ox, oy, scale, base, marker


def _parse(obs):
    decoded = _board(obs)
    if decoded is None:
        return None
    w, h, ox, oy, scale, base, marker = decoded

    targets = []
    for y in range(h):
        for x in range(w):
            if base[y, x] == 11 or marker[y, x] == 11:
                targets.append((x, y))

    axes = []
    # A mirror is a long horizontal or vertical line of color 10.
    for x in range(w):
        if int(np.count_nonzero(base[:, x] == 10)) >= max(5, h // 2):
            vals = [int(marker[y, x]) for y in range(h)
                    if base[y, x] == 10]
            selected = vals.count(0) > 0
            fixed = vals.count(10) > vals.count(9) + vals.count(0)
            axes.append({"kind": "v", "pos": x, "fixed": fixed,
                         "selected": selected})
    for y in range(h):
        if int(np.count_nonzero(base[y, :] == 10)) >= max(5, w // 2):
            vals = [int(marker[y, x]) for x in range(w)
                    if base[y, x] == 10]
            selected = vals.count(0) > 0
            fixed = vals.count(10) > vals.count(9) + vals.count(0)
            axes.append({"kind": "h", "pos": y, "fixed": fixed,
                         "selected": selected})

    # Separate real pieces from their color-4 reflected images.
    seen = set()
    pieces = []
    for y in range(h):
        for x in range(w):
            if base[y, x] != 5 or (x, y) in seen:
                continue
            q = deque([(x, y)])
            seen.add((x, y))
            cells = []
            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < w and 0 <= ny < h
                                and (nx, ny) not in seen
                                and base[ny, nx] == 5):
                            seen.add((nx, ny))
                            q.append((nx, ny))

            x0 = min(p[0] for p in cells)
            y0 = min(p[1] for p in cells)
            x1 = max(p[0] for p in cells)
            y1 = max(p[1] for p in cells)
            local = tuple(sorted((cx - x0, cy - y0) for cx, cy in cells))
            marks = [int(marker[cy, cx]) for cx, cy in cells]
            pieces.append({
                "x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
                "cells": local, "selected": marks.count(0) > 0,
                "fixed": marks.count(10) > 0 and marks.count(0) == 0,
                "allow_v": True, "allow_h": True,
            })

    # Infer one-axis-only pieces when a direct reflection is visibly absent.
    for p in pieces:
        real = {(p["x"] + dx, p["y"] + dy) for dx, dy in p["cells"]}
        for axis in axes:
            tested = hits = 0
            for x, y in real:
                if axis["kind"] == "v":
                    rx, ry = 2 * axis["pos"] - x, y
                else:
                    rx, ry = x, 2 * axis["pos"] - y
                if not (0 <= rx < w and 0 <= ry < h):
                    continue
                if base[ry, rx] == 5:
                    continue
                tested += 1
                if base[ry, rx] == 4:
                    hits += 1
            if tested >= 2 and hits == 0:
                if axis["kind"] == "v":
                    p["allow_v"] = False
                else:
                    p["allow_h"] = False

    return {
        "w": w, "h": h, "ox": ox, "oy": oy, "scale": scale,
        "base": base, "targets": targets, "axes": axes, "pieces": pieces,
    }


def _signature(board):
    return tuple(sorted(p["cells"] for p in board["pieces"]))


def _placement_masks(piece, targets, w, h, vertical, horizontal):
    """Map every useful translation to the targets covered there."""
    masks = {(piece["x"], piece["y"]): 0}
    for i, (tx, ty) in enumerate(targets):
        xs = [tx]
        ys = [ty]
        if vertical is not None and piece["allow_v"]:
            rx = 2 * vertical - tx
            if rx != tx:
                xs.append(rx)
        if horizontal is not None and piece["allow_h"]:
            ry = 2 * horizontal - ty
            if ry != ty:
                ys.append(ry)
        bit = 1 << i
        for sx in xs:
            for sy in ys:
                for dx, dy in piece["cells"]:
                    px, py = sx - dx, sy - dy
                    if 0 <= px <= w - piece["w"] and 0 <= py <= h - piece["h"]:
                        key = (px, py)
                        masks[key] = masks.get(key, 0) | bit
    return masks


def _solve(board):
    w, h = board["w"], board["h"]
    targets, axes, pieces = board["targets"], board["axes"], board["pieces"]
    if not pieces or not targets:
        return None

    full = (1 << len(targets)) - 1
    axis_ranges = []
    for a in axes:
        limit = w if a["kind"] == "v" else h
        axis_ranges.append([a["pos"]] if a["fixed"] else list(range(limit)))

    combinations = list(product(*axis_ranges)) if axes else [()]
    combinations.sort(key=lambda c: sum(
        abs(c[i] - axes[i]["pos"]) for i in range(len(axes))
    ))

    best = None
    best_cost = 10 ** 9
    partial = None
    partial_rank = (-1, -10 ** 9)
    node_count = 0
    branch_count = 0

    for combo in combinations:
        axis_cost = 0
        vertical = horizontal = None
        for i, a in enumerate(axes):
            delta = abs(combo[i] - a["pos"])
            if delta:
                axis_cost += delta + (0 if a["selected"] else 1)
            if a["kind"] == "v":
                vertical = combo[i]
            else:
                horizontal = combo[i]
        if axis_cost >= best_cost:
            continue

        options = []
        baseline = []
        reachable = 0

        for p in pieces:
            masks = _placement_masks(p, targets, w, h, vertical, horizontal)
            current = masks.get((p["x"], p["y"]), 0)
            baseline.append(current)

            if p["fixed"]:
                opts = [(0, p["x"], p["y"], current)]
            else:
                # For identical coverage, only the cheapest translation matters.
                by_mask = {}
                for (px, py), mask in masks.items():
                    if not mask and (px, py) != (p["x"], p["y"]):
                        continue
                    dist = abs(px - p["x"]) + abs(py - p["y"])
                    cost = dist + (1 if dist and not p["selected"] else 0)
                    old = by_mask.get(mask)
                    item = (cost, px, py, mask)
                    if old is None or item[:3] < old[:3]:
                        by_mask[mask] = item
                opts = sorted(by_mask.values())

            options.append(opts)
            for item in opts:
                reachable |= item[3]

        baseline_union = 0
        for m in baseline:
            baseline_union |= m

        # Keep a useful partial plan as a fallback.
        for i, opts in enumerate(options):
            other = 0
            for j, m in enumerate(baseline):
                if j != i:
                    other |= m
            for cost, px, py, mask in opts:
                cover = (other | mask).bit_count()
                rank = (cover, -(axis_cost + cost))
                if rank > partial_rank:
                    locs = [(p["x"], p["y"]) for p in pieces]
                    locs[i] = (px, py)
                    partial_rank = rank
                    partial = (combo, locs)

        if reachable != full:
            continue

        buckets = []
        for opts in options:
            per_target = [[] for _ in targets]
            for item in opts:
                mask = item[3]
                while mask:
                    bit = mask & -mask
                    per_target[bit.bit_length() - 1].append(item)
                    mask -= bit
            buckets.append(per_target)

        chosen = [None] * len(pieces)

        def search(used, fixed_cover, cost):
            nonlocal best, best_cost, node_count, branch_count
            node_count += 1
            if node_count > 150000 or branch_count > 1800000:
                return
            if cost >= best_cost:
                return

            cover = fixed_cover
            for j in range(len(pieces)):
                if not (used >> j) & 1:
                    cover |= baseline[j]
            missing = full & ~cover
            if not missing:
                locs = []
                for j, p in enumerate(pieces):
                    if chosen[j] is None:
                        locs.append((p["x"], p["y"]))
                    else:
                        locs.append((chosen[j][1], chosen[j][2]))
                best_cost = cost
                best = (combo, locs)
                return

            # Branch on the uncovered target with the fewest possible placements.
            target_index = -1
            min_count = 10 ** 9
            bits = missing
            while bits:
                bit = bits & -bits
                k = bit.bit_length() - 1
                count = 0
                for j in range(len(pieces)):
                    if not (used >> j) & 1:
                        count += len(buckets[j][k])
                if count < min_count:
                    min_count = count
                    target_index = k
                bits -= bit
            if min_count == 0:
                return

            candidates = []
            for j in range(len(pieces)):
                if not (used >> j) & 1:
                    for item in buckets[j][target_index]:
                        if cost + item[0] < best_cost:
                            candidates.append((item[0], j, item))
            candidates.sort(key=lambda z: z[0])

            for extra, j, item in candidates:
                branch_count += 1
                if branch_count > 1800000:
                    return
                chosen[j] = item
                search(used | (1 << j), fixed_cover | item[3], cost + extra)
                chosen[j] = None

        search(0, 0, axis_cost)
        if node_count > 150000 or branch_count > 1800000:
            break

    if best is not None:
        return best

    # Only use the fallback if it improves on the present covered target count.
    current_covered = sum(
        1 for x, y in targets if board["base"][y, x] in (4, 5)
    )
    if partial is not None and partial_rank[0] > current_covered:
        return partial
    return None


def _make_queue(board, solution):
    combo, locations = solution
    axes, pieces = board["axes"], board["pieces"]
    scale, ox, oy = board["scale"], board["ox"], board["oy"]
    w, h = board["w"], board["h"]

    def click(x, y):
        return ["ACTION6", oy + y * scale + scale // 2,
                ox + x * scale + scale // 2, False]

    selected = None
    for i, a in enumerate(axes):
        if a["selected"]:
            selected = ("a", i)
            break
    if selected is None:
        for i, p in enumerate(pieces):
            if p["selected"]:
                selected = ("p", i)
                break

    queue = []
    axis_positions = [a["pos"] for a in axes]
    piece_positions = [(p["x"], p["y"]) for p in pieces]

    # Move mirrors first. If a mirror rotates a piece, act() replans.
    for i, a in enumerate(axes):
        dst = combo[i]
        if dst == axis_positions[i]:
            continue
        if selected != ("a", i):
            occupied = set()
            for p, (px, py) in zip(pieces, piece_positions):
                occupied.update((px + dx, py + dy) for dx, dy in p["cells"])
            point = None
            if a["kind"] == "v":
                for y in range(h):
                    if (axis_positions[i], y) not in occupied and not any(
                        j != i and axes[j]["kind"] == "h"
                        and axis_positions[j] == y for j in range(len(axes))
                    ):
                        point = (axis_positions[i], y)
                        break
            else:
                for x in range(w):
                    if (x, axis_positions[i]) not in occupied and not any(
                        j != i and axes[j]["kind"] == "v"
                        and axis_positions[j] == x for j in range(len(axes))
                    ):
                        point = (x, axis_positions[i])
                        break
            if point is None:
                point = ((axis_positions[i], 0) if a["kind"] == "v"
                         else (0, axis_positions[i]))
            queue.append(click(*point))
            selected = ("a", i)

        delta = dst - axis_positions[i]
        if a["kind"] == "v":
            action = "ACTION4" if delta > 0 else "ACTION3"
        else:
            action = "ACTION2" if delta > 0 else "ACTION1"
        for _ in range(abs(delta)):
            queue.append([action, None, None, True])
        axis_positions[i] = dst

    for i, p in enumerate(pieces):
        px, py = piece_positions[i]
        dx, dy = locations[i][0] - px, locations[i][1] - py
        if dx == 0 and dy == 0:
            continue
        if selected != ("p", i):
            occupied = set()
            for j, other in enumerate(pieces):
                if i == j:
                    continue
                qx, qy = piece_positions[j]
                occupied.update((qx + x, qy + y) for x, y in other["cells"])
            choices = [(px + x, py + y) for x, y in p["cells"]
                       if (px + x, py + y) not in occupied]
            if not choices:
                choices = [(px + x, py + y) for x, y in p["cells"]]
            queue.append(click(*choices[0]))
            selected = ("p", i)
        for _ in range(abs(dx)):
            queue.append(["ACTION4" if dx > 0 else "ACTION3",
                          None, None, False])
        for _ in range(abs(dy)):
            queue.append(["ACTION2" if dy > 0 else "ACTION1",
                          None, None, False])
        piece_positions[i] = locations[i]

    return queue


def act(obs, available, memory):
    board = _parse(obs)
    if board is None:
        return ("ACTION5" if "ACTION5" in available else available[0], None, None)

    signature = _signature(board)
    if memory.get("last_axis") and memory.get("signature") != signature:
        memory.pop("queue", None)
    memory["signature"] = signature
    memory["last_axis"] = False

    queue = memory.get("queue")
    if not queue:
        solution = _solve(board)
        queue = _make_queue(board, solution) if solution is not None else []
        memory["queue"] = queue

    if queue:
        action, row, col, axis_move = queue.pop(0)
        memory["last_axis"] = axis_move
        if action in available:
            return action, row, col

    # If parsing or the search is inconclusive, try a deterministic local move.
    pieces = board["pieces"]
    targets = board["targets"]
    if pieces and targets:
        p = next((p for p in pieces if p["selected"]), pieces[0])
        uncovered = [(x, y) for x, y in targets
                     if board["base"][y, x] not in (4, 5)]
        if uncovered:
            px = p["x"] + p["w"] // 2
            py = p["y"] + p["h"] // 2
            tx, ty = min(uncovered, key=lambda t: abs(t[0] - px) + abs(t[1] - py))
            if not p["selected"] and "ACTION6" in available:
                x, y = p["cells"][0]
                return ("ACTION6",
                        board["oy"] + (p["y"] + y) * board["scale"] + board["scale"] // 2,
                        board["ox"] + (p["x"] + x) * board["scale"] + board["scale"] // 2)
            if tx != px:
                action = "ACTION4" if tx > px else "ACTION3"
            else:
                action = "ACTION2" if ty > py else "ACTION1"
            if action in available:
                return action, None, None

    for action in ("ACTION1", "ACTION5", "ACTION2", "RESET"):
        if action in available:
            return action, None, None
    return available[0], None, None

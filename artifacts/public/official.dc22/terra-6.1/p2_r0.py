from collections import deque


DIRECTIONS = (
    ("ACTION1", 0, -2),
    ("ACTION2", 0, 2),
    ("ACTION3", -2, 0),
    ("ACTION4", 2, 0),
)

CRANE_SHAPES = (
    (
        "...#####",
        "...#...#",
        ".####...",
        "####....",
        "####....",
        ".####...",
        "...#...#",
        "...#####",
    ),
    (
        "........",
        "...#####",
        ".####..#",
        "####....",
        "####....",
        ".####..#",
        "...#####",
        "........",
    ),
)


def components(obs, color, x0, x1, y0, y1):
    seen = set()
    result = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            if int(obs[y, x]) != color or (x, y) in seen:
                continue
            seen.add((x, y))
            stack = [(x, y)]
            pixels = []
            while stack:
                a, b = stack.pop()
                pixels.append((a, b))
                for u, v in ((a - 1, b), (a + 1, b),
                             (a, b - 1), (a, b + 1)):
                    if (x0 <= u < x1 and y0 <= v < y1
                            and (u, v) not in seen
                            and int(obs[v, u]) == color):
                        seen.add((u, v))
                        stack.append((u, v))
            xx = [p[0] for p in pixels]
            yy = [p[1] for p in pixels]
            result.append({
                "pixels": pixels,
                "x0": min(xx), "x1": max(xx),
                "y0": min(yy), "y1": max(yy),
                "w": max(xx) - min(xx) + 1,
                "h": max(yy) - min(yy) + 1,
                "n": len(pixels),
            })
    return result


def component_point(c):
    cx = (c["x0"] + c["x1"]) / 2.0
    cy = (c["y0"] + c["y1"]) / 2.0
    x, y = min(c["pixels"],
               key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    return [int(y), int(x)]


def bounds(obs):
    y0 = 0
    while y0 < 62 and all(int(obs[y0, x]) == 3 for x in range(64)):
        y0 += 1
    y1 = 63
    for y in range(y0 + 1, 63):
        if all(int(obs[y, x]) == 3 for x in range(64)):
            y1 = y
            break
    height = max(1, y1 - y0)
    bx = 64
    for x in range(18, 61):
        dark = sum(int(obs[y, x]) in (0, 5) for y in range(y0, y1))
        black = sum(int(obs[y, x]) == 5 for y in range(y0, y1))
        if dark >= 0.93 * height and black >= 0.25 * height:
            bx = x
            break
    return y0, y1, bx


def find_player(obs, y0, y1, bx):
    for y in range(y0, min(y1 - 1, 62)):
        for x in range(max(0, bx - 1)):
            if (int(obs[y, x]) == 14 and int(obs[y, x + 1]) == 14
                    and int(obs[y + 1, x]) == 14
                    and int(obs[y + 1, x + 1]) == 14):
                return (x, y)
    return None


def panel_info(obs, y0, y1, bx):
    buttons = {}
    comps = {}
    if bx >= 64:
        return buttons, comps
    for color in (2, 6, 8, 9, 10, 11, 15):
        cs = components(obs, color, bx, 64, y0, y1)
        comps[color] = cs
        if color in (6, 8, 9, 15):
            choices = [c for c in cs if c["w"] >= 7 and c["h"] >= 4]
        elif color in (2, 11):
            choices = [c for c in cs if c["w"] >= 4 and c["h"] >= 4]
        else:
            choices = []
        if choices:
            buttons[color] = component_point(max(choices, key=lambda c: c["n"]))
    return buttons, comps


def signature(obs, x, y):
    if not (0 <= x < 63 and 0 <= y < 63):
        return (-1, -1, -1, -1)
    return (int(obs[y, x]), int(obs[y, x + 1]),
            int(obs[y + 1, x]), int(obs[y + 1, x + 1]))


def failure_key(x, y, action, sig):
    return "%d,%d,%s,%s" % (
        x, y, action, ",".join(str(int(v)) for v in sig)
    )


def crane_controls(panel_components):
    nines = [c for c in panel_components.get(9, [])
             if 3 <= c["w"] <= 5 and 3 <= c["h"] <= 5]
    tens = [c for c in panel_components.get(10, [])
            if 3 <= c["w"] <= 5 and 3 <= c["h"] <= 5]
    grabs = [c for c in panel_components.get(8, [])
             if c["w"] >= 7 and c["h"] == 3]
    if len(nines) < 2 or len(tens) < 2 or not grabs:
        return None
    nines.sort(key=lambda c: c["x0"])
    tens.sort(key=lambda c: c["x0"])
    return {
        "up": component_point(nines[0]),
        "down": component_point(nines[-1]),
        "left": component_point(tens[0]),
        "right": component_point(tens[-1]),
        "grab": component_point(max(grabs, key=lambda c: c["n"])),
    }


def find_crane(obs, player, y0, y1, bx, controls, memory):
    if controls is None:
        return None

    origin = memory.get("crane_origin")
    if origin is None:
        zeros = [(x, y) for y in range(y0, y1)
                 for x in range(0, bx)
                 if int(obs[y, x]) == 0]
        if not zeros:
            return None
        minx = min(x for x, y in zeros)
        nearby = [(x, y) for x, y in zeros if x <= minx + 22]
        if len(nearby) < 40:
            return None
        bottom = max(y for x, y in nearby)
        origin = [minx - 1, bottom - 5]
        if origin[0] < 0 or origin[1] - 12 < y0:
            return None
        memory["crane_origin"] = origin

    ox, oy = origin
    best = None
    for ix in range(4):
        for iy in range(4):
            if ix != 0 and iy not in (0, 3):
                continue
            x, y = ox + 4 * ix, oy - 4 * iy
            if x < 0 or y < y0 or x + 8 > bx or y + 8 > y1:
                continue
            for kind, shape in enumerate(CRANE_SHAPES):
                matched = 0
                missing = 0
                extra = 0
                for dy in range(8):
                    for dx in range(8):
                        v = int(obs[y + dy, x + dx])
                        expected = shape[dy][dx] == "#"
                        if expected:
                            if v == 8:
                                matched += 1
                            elif v != 14:
                                missing += 1
                        elif v == 8:
                            extra += 1
                score = 3 * matched - 4 * missing - 2 * extra
                if matched >= 18:
                    candidate = (score, matched, -missing, ix, iy, kind)
                    if best is None or candidate > best:
                        best = candidate
    if best is None:
        return None

    _, _, _, ix, iy, kind = best
    cx, cy = ox + 4 * ix, oy - 4 * iy

    obj_best = None
    for y in range(y0, max(y0, y1 - 11)):
        if (y - player[1]) % 2:
            continue
        for x in range(player[0] % 2, max(0, bx - 5), 2):
            matched = 0
            missing = 0
            extra = 0
            for dy in range(12):
                for dx in range(6):
                    expected = dx >= 2 or (dx == 0 and 4 <= dy <= 7) or (
                        dx == 1 and 5 <= dy <= 6
                    )
                    v = int(obs[y + dy, x + dx])
                    if expected:
                        if v == 12:
                            matched += 1
                        elif v not in (8, 14):
                            missing += 1
                    elif v == 12:
                        extra += 1
            score = 3 * matched - 3 * missing - extra
            if matched >= 25:
                candidate = (score, matched, -missing, x, y)
                if obj_best is None or candidate > obj_best:
                    obj_best = candidate
    if obj_best is None:
        saved = memory.get("crane_object")
        if saved is None:
            return None
        obj = tuple(saved)
    else:
        obj = (obj_best[3], obj_best[4])

    grabbed = kind == 1 and obj == (cx + 6, cy - 2)
    if grabbed:
        obj = (cx + 6, cy - 2)
    memory["crane_object"] = [obj[0], obj[1]]

    return {
        "origin": (ox, oy),
        "ix": ix,
        "iy": iy,
        "grabbed": int(grabbed),
        "object": obj,
        "controls": controls,
        "current": (cx, cy),
    }


def find_lanes(obs, cells, tile, buttons, memory):
    saved = memory.get("lanes", {})
    laser = saved.get("laser")
    slider = saved.get("slider")

    if laser is None and 2 in buttons:
        candidates = []
        ys = sorted(set(y for x, y in cells))
        for y in ys:
            row = sorted(x for x, yy in cells if yy == y)
            i = 0
            while i < len(row):
                vals = tile[(row[i], y)]
                if not all(v in (1, 5, 12, 14) for v in vals):
                    i += 1
                    continue
                run = []
                while i < len(row):
                    xx = row[i]
                    vv = tile[(xx, y)]
                    if not all(v in (1, 5, 12, 14) for v in vv):
                        break
                    run.append(xx)
                    i += 1
                if len(run) >= 8:
                    for j in range(len(run) - 7):
                        segment = run[j:j + 8]
                        if segment[-1] - segment[0] != 14:
                            continue
                        colors = [v for xx in segment for v in tile[(xx, y)]]
                        if 5 in colors or colors.count(12) >= 24:
                            candidates.append((segment[0], y))
        if candidates:
            laser = list(candidates[0])

    if slider is None and 11 in buttons:
        runs = []
        ys = sorted(set(y for x, y in cells))
        for y in ys:
            row = sorted(x for x, yy in cells
                         if yy == y and all(v == 1 for v in tile[(x, y)]))
            start = 0
            while start < len(row):
                end = start + 1
                while end < len(row) and row[end] == row[end - 1] + 2:
                    end += 1
                group = row[start:end]
                if len(group) == 3:
                    runs.append((group[0], y))
                start = end
        if runs:
            if laser is not None:
                near = [(x, y) for x, y in runs
                        if laser[0] <= x <= laser[0] + 10
                        and y != laser[1]]
                if near:
                    x, y = min(near, key=lambda p: abs(p[1] - laser[1]))
                    slider = [laser[0], y]
            if slider is None:
                slider = list(runs[0])

    if laser is not None:
        saved["laser"] = [int(laser[0]), int(laser[1])]
    if slider is not None:
        saved["slider"] = [int(slider[0]), int(slider[1])]
    memory["lanes"] = saved

    laser_phase = 0
    if laser is not None:
        x, y = laser
        vals = [tile.get((x + 2 * i, y), (-1,) * 4) for i in range(8)]
        if sum(v.count(12) for v in vals) >= 20:
            possible = [4]
        else:
            left = 0
            for v in vals:
                if 1 in v:
                    left += 1
                else:
                    break
            right = 0
            for v in reversed(vals):
                if 1 in v:
                    right += 1
                else:
                    break
            n = min(left, right, 3)
            possible = [0] if n == 0 else [n, 8 - n]
        old = memory.get("laser_phase")
        laser_phase = old if old in possible else possible[0]
        memory["laser_phase"] = laser_phase

    slider_phase = 0
    if slider is not None:
        x, y = slider
        positions = []
        for i in range(8):
            vals = tile.get((x + 2 * i, y), (-1,) * 4)
            if 1 in vals or 12 in vals:
                positions.append(i)
        if positions:
            guessed = max(0, min(5, min(positions)))
            old = memory.get("slider_phase")
            if old is not None and old in range(6):
                if set(positions).issubset(set(range(old, old + 3))):
                    guessed = old
            slider_phase = guessed
        else:
            slider_phase = int(memory.get("slider_phase", 0)) % 6
        memory["slider_phase"] = slider_phase

    return laser, laser_phase, slider, slider_phase


def build_model(obs, player, y0, y1, bx, buttons, crane, memory):
    px, py = player
    xs = list(range(px % 2, max(px % 2, bx - 1), 2))
    first_y = y0 + ((py - y0) % 2)
    ys = list(range(first_y, max(first_y, y1 - 1), 2))
    cells = {(x, y) for y in ys for x in xs}
    tile = {p: signature(obs, *p) for p in cells}

    key_cells = set()
    for y in ys:
        if y + 1 >= y1:
            continue
        for x in xs:
            if x + 3 >= bx:
                continue
            if (int(obs[y, x]) != 15
                    and int(obs[y, x + 1]) == 15
                    and int(obs[y, x + 2]) == 15
                    and int(obs[y, x + 3]) != 15
                    and all(int(obs[y + 1, x + k]) == 15
                            for k in range(4))):
                key_cells.add((x, y))
                key_cells.add((x + 2, y))

    support = set()
    block = set()
    yellow = {}
    pink = {}
    bridge_pixels = set()
    pivots = set()
    teleports = {}
    goals = set()

    for p, values in tile.items():
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        x, y = p

        if 2 in counts or 1 in counts:
            support.add(p)
        if 12 in counts and crane is None:
            support.add(p)
        if 6 in counts and 7 in counts:
            support.add(p)
            teleports[p] = values[0]
        if counts.get(11, 0) >= 3:
            support.add(p)
            goals.add(p)
        if p in key_cells:
            support.add(p)

        # In the rail mechanism, the large black rail sprites have the
        # game's "ignore" tag and do not block the player.
        if 0 in counts and crane is None:
            block.add(p)

        if 9 in counts:
            yellow[p] = counts[9] >= 3
        if 15 in counts and p not in key_cells:
            pink[p] = counts[15] >= 3

        in_crane = False
        if crane is not None:
            cx, cy = crane["current"]
            in_crane = (cx <= x < cx + 8 and cy <= y < cy + 8)
        if not in_crane and (8 in counts or 13 in counts):
            bridge_pixels.add(p)
        if 13 in counts:
            pivots.add(p)

    groups = []
    unvisited = set(bridge_pixels)
    while unvisited:
        first = unvisited.pop()
        group = {first}
        stack = [first]
        while stack:
            x, y = stack.pop()
            for p in ((x - 2, y), (x + 2, y),
                      (x, y - 2), (x, y + 2)):
                if p in unvisited:
                    unvisited.remove(p)
                    group.add(p)
                    stack.append(p)
        pp = group & pivots
        if pp:
            groups.append((group, pp))
        else:
            block.update(group)

    bridge_now = set()
    bridge_rotated = set()
    for group, pp in groups:
        minx = min(x for x, y in pp)
        maxx = max(x for x, y in pp)
        miny = min(y for x, y in pp)
        maxy = max(y for x, y in pp)
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        gx0 = min(x for x, y in group)
        gx1 = max(x for x, y in group)
        gy0 = min(y for x, y in group)
        gy1 = max(y for x, y in group)
        horizontal = gx1 - gx0 >= gy1 - gy0
        group = set(group)
        if gx0 <= px <= gx1 and gy0 <= py <= gy1:
            group.add(player)

        rotated = set()
        for x, y in group:
            if horizontal:
                nx = cx - (y - cy)
                ny = cy + (x - cx)
            else:
                nx = cx + (y - cy)
                ny = cy - (x - cx)
            if abs(nx - round(nx)) < 0.01 and abs(ny - round(ny)) < 0.01:
                p = (int(round(nx)), int(round(ny)))
                if p in cells:
                    rotated.add(p)
        bridge_now.update(group)
        bridge_rotated.update(rotated)

    laser, laser_phase, slider, slider_phase = find_lanes(
        obs, cells, tile, buttons, memory
    )
    laser_cells = set()
    slider_cells = set()
    if laser is not None:
        laser_cells = {(laser[0] + 2 * i, laser[1]) for i in range(8)}
    if slider is not None:
        slider_cells = {(slider[0] + 2 * i, slider[1]) for i in range(8)}
    support.difference_update(laser_cells | slider_cells)
    block.difference_update(laser_cells | slider_cells)

    on_object = False
    if crane is not None:
        x, y = crane["object"]
        on_object = x <= px < x + 6 and y <= py < y + 12

    if (player not in support and player not in yellow
            and player not in pink and player not in bridge_now
            and player not in laser_cells and player not in slider_cells
            and not on_object):
        neighbors = [(px - 2, py), (px + 2, py),
                     (px, py - 2), (px, py + 2)]
        inferred = False
        for dynamic in (yellow, pink):
            values = [dynamic[p] for p in neighbors if p in dynamic]
            if len(values) >= 2 and all(v == values[0] for v in values):
                dynamic[player] = values[0]
                inferred = True
                break
        if not inferred:
            support.add(player)

    supports = []
    blocks = []
    for mask in range(8):
        s = set(support)
        b = set(block)
        for p, full in yellow.items():
            if full != bool(mask & 2):
                s.add(p)
            else:
                b.add(p)
        for p, full in pink.items():
            if full != bool(mask & 4):
                s.add(p)
            else:
                b.add(p)
        if mask & 1:
            s.update(bridge_rotated)
        else:
            s.update(bridge_now)
        supports.append(s)
        blocks.append(b)

    return {
        "cells": cells,
        "tile": tile,
        "supports": supports,
        "blocks": blocks,
        "goals": goals,
        "keys": key_cells,
        "teleports": teleports,
        "yellow": yellow,
        "pink": pink,
        "bridges": groups,
        "laser": laser,
        "laser_phase": laser_phase,
        "slider": slider,
        "slider_phase": slider_phase,
        "crane": crane,
    }


def teleport_destination(pos, teleports):
    if pos not in teleports or len(teleports) < 2:
        return pos
    candidates = [p for p in teleports
                  if p != pos and teleports[p] != teleports[pos]]
    if not candidates:
        candidates = [p for p in teleports if p != pos]
    return min(candidates, key=lambda p: (p[1], p[0])) if candidates else pos


def search(model, player, buttons, available, memory):
    supports = model["supports"]
    blocks = model["blocks"]
    goals = model["goals"]
    keys = model["keys"]
    teleports = model["teleports"]
    laser = model["laser"]
    slider = model["slider"]
    crane = model["crane"]
    failed = set(memory.get("failed_moves", []))
    visits = memory.get("visits", {})

    initially_unlocked = int(15 in buttons)
    key_stage = bool(crane is not None and keys and not initially_unlocked)

    if crane is None:
        ci, cj, grabbed = 0, 0, 0
    else:
        ci, cj, grabbed = crane["ix"], crane["iy"], crane["grabbed"]

    start = (player[0], player[1], 0, initially_unlocked,
             model["laser_phase"], model["slider_phase"],
             ci, cj, grabbed)

    def object_position(ix, iy, held):
        if crane is None:
            return None
        if held:
            ox, oy = crane["origin"]
            return (ox + 4 * ix + 6, oy - 4 * iy - 2)
        return crane["object"]

    def object_support(pos, ix, iy, held):
        obj = object_position(ix, iy, held)
        if obj is None:
            return False
        dx = pos[0] - obj[0]
        dy = pos[1] - obj[1]
        if not (0 <= dy < 12):
            return False
        if dx in (2, 4):
            return True
        return dx == 0 and dy in (4, 6)

    def dynamic_support(pos, lp, sp):
        if laser is not None and pos[1] == laser[1]:
            d = pos[0] - laser[0]
            if 0 <= d <= 14 and d % 2 == 0:
                i = d // 2
                if lp == 4:
                    return True
                n = min(lp, 8 - lp)
                if i < n or i >= 8 - n:
                    return True
        if slider is not None and pos[1] == slider[1]:
            d = pos[0] - slider[0]
            if 0 <= d <= 14 and d % 2 == 0:
                i = d // 2
                if sp <= i < sp + 3:
                    return True
        return False

    def is_supported(pos, mask, lp, sp, ix, iy, held):
        return (pos in supports[mask]
                or dynamic_support(pos, lp, sp)
                or object_support(pos, ix, iy, held))

    def crane_collision(pos, ix, iy, held):
        if crane is None:
            return False
        ox, oy = crane["origin"]
        cx = ox + 4 * ix
        cy = oy - 4 * iy
        shape = CRANE_SHAPES[1 if held else 0]
        x, y = pos
        for dy in range(2):
            for dx in range(2):
                xx, yy = x + dx - cx, y + dy - cy
                if 0 <= xx < 8 and 0 <= yy < 8:
                    if shape[yy][xx] == "#":
                        return True
        return False

    switches = []
    if 8 in buttons and model["bridges"]:
        switches.append((8, "bridge"))
    if 9 in buttons and model["yellow"]:
        switches.append((9, "yellow"))
    if model["pink"] and (15 in buttons or keys) and not key_stage:
        switches.append((15, "pink"))
    if 6 in buttons and teleports:
        switches.append((6, "teleport"))
    if not key_stage:
        if 2 in buttons and laser is not None:
            switches.append((2, "laser"))
        if 11 in buttons and slider is not None:
            switches.append((11, "slider"))

    queue = deque([start])
    parent = {start: None}
    via = {}
    depth = {start: 0}
    found = None
    best = None
    best_score = float("inf")
    limit = 300000

    def add(state, previous, action):
        if state not in parent:
            parent[state] = previous
            via[state] = action
            depth[state] = depth[previous] + 1
            queue.append(state)

    while queue and len(parent) < limit:
        state = queue.popleft()
        x, y, mask, unlocked, lp, sp, ix, iy, held = state
        pos = (x, y)

        if key_stage:
            if pos in keys and state != start:
                found = state
                break
        elif pos in goals and state != start:
            found = state
            break

        targets = keys if keys and not unlocked else goals
        if state != start and targets:
            dist = min(abs(x - gx) + abs(y - gy) for gx, gy in targets) / 2
            count = visits.get("%d,%d" % pos, 0)
            score = dist + 5 * count + 0.08 * depth[state]
            if score < best_score:
                best_score = score
                best = state

        for action, dx, dy in DIRECTIONS:
            if action not in available:
                continue
            np = (x + dx, y + dy)
            if np not in model["cells"]:
                continue
            if not is_supported(np, mask, lp, sp, ix, iy, held):
                continue
            if np in blocks[mask] or crane_collision(np, ix, iy, held):
                continue
            if (mask == 0 and lp == model["laser_phase"]
                    and sp == model["slider_phase"]
                    and (crane is None or
                         (ix, iy, held) == (ci, cj, grabbed))):
                sig = model["tile"].get(np, (-1, -1, -1, -1))
                if failure_key(x, y, action, sig) in failed:
                    continue
            ns = (np[0], np[1], mask, int(unlocked or np in keys),
                  lp, sp, ix, iy, held)
            add(ns, state, (action, None))

        if "ACTION6" not in available:
            continue

        for color, kind in switches:
            if kind == "pink" and not unlocked:
                continue
            nm, nlp, nsp = mask, lp, sp
            np = pos
            if kind == "bridge":
                nm ^= 1
            elif kind == "yellow":
                nm ^= 2
            elif kind == "pink":
                nm ^= 4
            elif kind == "teleport":
                np = teleport_destination(pos, teleports)
                if np == pos:
                    continue
            elif kind == "laser":
                nlp = (lp + 1) % 8
            elif kind == "slider":
                nsp = (sp + 1) % 6
            if not is_supported(np, nm, nlp, nsp, ix, iy, held):
                continue
            ns = (np[0], np[1], nm, unlocked, nlp, nsp, ix, iy, held)
            add(ns, state, ("ACTION6", color))

        # During the key stage, search the crane's rail positions. Once the
        # key is collected, leave the crane fixed to keep the search small.
        if crane is not None and key_stage:
            controls = crane["controls"]
            for name, di, dj in (
                ("up", 0, 1), ("down", 0, -1),
                ("left", -1, 0), ("right", 1, 0)
            ):
                ni, nj = ix + di, iy + dj
                if not (0 <= ni <= 3 and 0 <= nj <= 3):
                    continue
                if ni != 0 and nj not in (0, 3):
                    continue
                if di and iy not in (0, 3):
                    continue
                if dj and ix != 0:
                    continue
                if not is_supported(pos, mask, lp, sp, ni, nj, held):
                    continue
                ns = (x, y, mask, unlocked, lp, sp, ni, nj, held)
                add(ns, state, ("ACTION6", controls[name]))

            if not held:
                ox, oy = crane["origin"]
                anchor_obj = (ox + 4 * ix + 6, oy - 4 * iy - 2)
                if anchor_obj == crane["object"]:
                    ns = (x, y, mask, unlocked, lp, sp, ix, iy, 1)
                    add(ns, state, ("ACTION6", controls["grab"]))

    target = found if found is not None else best
    if target is None:
        return None, False
    plan = []
    while parent[target] is not None:
        action, param = via[target]
        plan.append([action, param, target[0], target[1]])
        target = parent[target]
    plan.reverse()
    return plan, found is not None


def emit(obs, player, item, buttons, memory, available):
    action, param, ex, ey = item
    if action not in available:
        return None

    row = col = None
    if action == "ACTION6":
        if isinstance(param, (list, tuple)):
            row, col = int(param[0]), int(param[1])
        else:
            point = buttons.get(int(param))
            if point is None:
                return None
            row, col = point
            if param == 2:
                memory["laser_phase"] = (
                    int(memory.get("laser_phase", 0)) + 1
                ) % 8
            elif param == 11:
                memory["slider_phase"] = (
                    int(memory.get("slider_phase", 0)) + 1
                ) % 6

    sig = None
    for name, dx, dy in DIRECTIONS:
        if action == name:
            sig = signature(obs, player[0] + dx, player[1] + dy)
            break
    memory["previous"] = {
        "pos": [player[0], player[1]],
        "expected": [int(ex), int(ey)],
        "action": action,
        "signature": list(sig) if sig is not None else None,
    }
    return action, row, col


def act(obs, available, memory):
    y0, y1, bx = bounds(obs)
    player = find_player(obs, y0, y1, bx)
    buttons, panel_components = panel_info(obs, y0, y1, bx)

    if player is None:
        for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if action in available:
                return action, None, None
        return available[0], None, None

    visits = memory.setdefault("visits", {})
    key = "%d,%d" % player
    visits[key] = visits.get(key, 0) + 1

    previous = memory.pop("previous", None)
    if previous is not None:
        old = tuple(previous["pos"])
        expected = tuple(previous["expected"])
        if player != expected:
            memory["plan"] = []
            sig = previous.get("signature")
            if player == old and sig is not None:
                failures = memory.setdefault("failed_moves", [])
                k = failure_key(old[0], old[1], previous["action"], sig)
                if k not in failures:
                    failures.append(k)
                    if len(failures) > 300:
                        del failures[:100]

    plan = memory.get("plan", [])
    if plan:
        item = plan.pop(0)
        memory["plan"] = plan
        output = emit(obs, player, item, buttons, memory, available)
        if output is not None:
            return output
        memory["plan"] = []

    controls = crane_controls(panel_components)
    crane = find_crane(obs, player, y0, y1, bx, controls, memory)
    model = build_model(obs, player, y0, y1, bx, buttons, crane, memory)
    plan, solves = search(model, player, buttons, available, memory)

    if plan:
        item = plan.pop(0)
        memory["plan"] = plan
        output = emit(obs, player, item, buttons, memory, available)
        if output is not None:
            return output
        memory["plan"] = []

    failed = set(memory.get("failed_moves", []))
    for action, dx, dy in DIRECTIONS:
        p = (player[0] + dx, player[1] + dy)
        if (action in available and p in model["supports"][0]
                and p not in model["blocks"][0]):
            sig = model["tile"].get(p, (-1, -1, -1, -1))
            if failure_key(player[0], player[1], action, sig) not in failed:
                output = emit(obs, player, [action, None, p[0], p[1]],
                              buttons, memory, available)
                if output is not None:
                    return output

    for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        if action in available:
            return action, None, None
    if "ACTION6" in available and buttons:
        p = next(iter(buttons.values()))
        return "ACTION6", int(p[0]), int(p[1])
    return available[0], None, None

from collections import deque
import numpy as np


def components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=np.uint8)
    result = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            seen[y, x] = 1
            q = [(x, y)]
            for px, py in q:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = px + dx, py + dy
                    if (0 <= nx < w and 0 <= ny < h and
                            mask[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = 1
                        q.append((nx, ny))
            result.append(q)
    return result


def find_player(a):
    for y in range(62):
        for x in range(63):
            if (a[y, x] == 14 and a[y, x + 1] == 14 and
                    a[y + 1, x] == 14 and a[y + 1, x + 1] == 14):
                return (x, y)
    return None


def bounds(a):
    rows = [y for y in range(63) if np.count_nonzero(a[y] == 3) < 60]
    if not rows:
        return 0, 62, 38
    y0, y1 = min(rows), max(rows)
    height = y1 - y0 + 1
    cut = 64
    for x in range(20, 64):
        if np.count_nonzero(a[y0:y1 + 1, x] == 5) >= 0.82 * height:
            cut = max(2, x - 2)
            break
    return y0, y1, cut


def buttons_from_frame(a, cut, y0, y1):
    buttons = {}
    for color, name in ((8, "a"), (9, "b"), (6, "c"), (15, "d")):
        found = None
        for y in range(max(0, y0), min(59, y1 - 3) + 1):
            for x in range(max(0, cut), 52):
                if (np.all(a[y:y + 3, x + 3:x + 10] == color) and
                        np.all(a[y + 3:y + 5, x:x + 13] == color)):
                    found = (y + 1, x + 6)
                    break
            if found:
                break
        if found:
            buttons[name] = found

    for color, name in ((2, "e"), (11, "f")):
        mask = np.zeros_like(a, dtype=bool)
        mask[max(0, y0):min(63, y1 + 1), cut:] = (
            a[max(0, y0):min(63, y1 + 1), cut:] == color
        )
        candidates = []
        for comp in components(mask):
            if len(comp) < 12:
                continue
            xs = [p[0] for p in comp]
            ys = [p[1] for p in comp]
            if max(xs) - min(xs) <= 14 and max(ys) - min(ys) <= 10:
                candidates.append(comp)
        if candidates:
            comp = max(candidates, key=len)
            x, y = comp[len(comp) // 2]
            buttons[name] = (y, x)

    crane = None
    for y in range(max(0, y0), min(59, y1 - 3) + 1):
        for x in range(max(0, cut), 45):
            if all(
                np.all(a[y:y + 4, x + 5 * i:x + 5 * i + 4] == c)
                for i, c in enumerate((10, 9, 10, 9))
            ):
                crane = {
                    "left": (y + 1, x + 1),
                    "up": (y + 1, x + 6),
                    "right": (y + 1, x + 11),
                    "down": (y + 1, x + 16),
                }
                break
        if crane:
            break

    if crane:
        found = False
        for y in range(max(0, y0), min(61, y1 - 1) + 1):
            for x in range(max(0, cut), 56):
                if np.all(a[y:y + 3, x:x + 9] == 8):
                    crane["grab"] = (y + 1, x + 4)
                    found = True
                    break
            if found:
                break
        if not found:
            crane = None
    return buttons, crane


def find_bridges(a, cut, parity):
    mask = np.zeros_like(a, dtype=bool)
    mask[:, :cut] = a[:, :cut] == 13
    result = []
    for comp in components(mask):
        if len(comp) < 6:
            continue
        xs = [p[0] for p in comp]
        ys = [p[1] for p in comp]
        if max(xs) - min(xs) > 5 or max(ys) - min(ys) > 5:
            continue
        x, y = min(xs), min(ys)
        x -= (x - parity[0]) % 2
        y -= (y - parity[1]) % 2

        def count8(x0, x1, y0, y1):
            x0, x1 = max(0, x0), min(cut, x1)
            y0, y1 = max(0, y0), min(63, y1)
            if x1 <= x0 or y1 <= y0:
                return 0
            return int(np.count_nonzero(a[y0:y1, x0:x1] == 8))

        left = count8(x - 8, x, y, y + 4)
        right = count8(x + 4, x + 12, y, y + 4)
        up = count8(x, x + 4, y - 8, y)
        down = count8(x, x + 4, y + 4, y + 12)
        horizontal = left + right >= up + down
        two_sided = right >= 5 if horizontal else down >= 5
        length = 8 if two_sided else 6

        hs, vs = set(), set()
        for yy in range(y, y + 4, 2):
            for xx in range(x - length, x + 4 + (length if two_sided else 0), 2):
                hs.add((xx, yy))
        for xx in range(x, x + 4, 2):
            for yy in range(y - length, y + 4 + (length if two_sided else 0), 2):
                vs.add((xx, yy))
        result.append((hs, vs, horizontal))
    return result


def find_tool(a, cut):
    for y in range(52):
        for x in range(max(0, cut - 5)):
            core = a[y:y + 12, x + 2:x + 6]
            if core.shape != (12, 4):
                continue
            good = np.count_nonzero((core == 12) | (core == 14) | (core == 8))
            if good < 46:
                continue
            if (a[y + 5, x] == 12 and a[y + 6, x] == 12 and
                    a[y + 5, x + 1] == 12 and a[y + 6, x + 1] == 12):
                return (x, y)
    return None


def find_cursor(a, cut):
    patterns = [
        (
            [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
             (3, 1), (7, 1),
             (1, 2), (2, 2), (3, 2), (4, 2),
             (0, 3), (1, 3), (2, 3), (3, 3),
             (0, 4), (1, 4), (2, 4), (3, 4),
             (1, 5), (2, 5), (3, 5), (4, 5),
             (3, 6), (7, 6),
             (3, 7), (4, 7), (5, 7), (6, 7), (7, 7)],
            0
        ),
        (
            [(3, 1), (4, 1), (5, 1), (6, 1), (7, 1),
             (1, 2), (2, 2), (3, 2), (4, 2), (7, 2),
             (0, 3), (1, 3), (2, 3), (3, 3),
             (0, 4), (1, 4), (2, 4), (3, 4),
             (1, 5), (2, 5), (3, 5), (4, 5), (7, 5),
             (3, 6), (4, 6), (5, 6), (6, 6), (7, 6)],
            1
        ),
    ]
    best = None
    for y in range(56):
        for x in range(max(0, cut - 7)):
            for pts, grabbed in patterns:
                hits = sum(a[y + dy, x + dx] == 8 for dx, dy in pts)
                if hits < len(pts) - 2:
                    continue
                extra = int(np.count_nonzero(a[y:y + 8, x:x + 8] == 8)) - hits
                if extra > 4:
                    continue
                score = hits - 2 * max(0, extra)
                if best is None or score > best[0]:
                    best = (score, x, y, grabbed)
    return best


def find_moving(a, cut, buttons, parity):
    if "e" not in buttons:
        return None
    candidates = []
    for y in range(parity[1], 61, 2):
        for x in range(parity[0], max(parity[0], cut - 15), 2):
            tile = a[y:y + 2, x:x + 16]
            if tile.shape != (2, 16):
                continue
            if not np.all(np.isin(tile, (1, 5, 12, 14))):
                continue
            if x > 0 and a[y, x - 1] in (1, 5, 12):
                continue
            if x + 16 < cut and a[y, x + 16] in (1, 5, 12):
                continue
            candidates.append((x, y))
    if not candidates:
        return None

    for x, y in candidates:
        for fy in range(max(parity[1], y - 16), min(61, y + 17), 2):
            if fy == y:
                continue
            row = a[fy, x:x + 16]
            if len(row) != 16:
                continue
            for offset in (0, 2, 4, 6, 8, 10):
                color = int(row[offset])
                if color not in (1, 12):
                    continue
                if np.all(row[offset:offset + 6] == color) and "f" in buttons:
                    erow = a[y, x:x + 16]
                    ep = 4 if np.count_nonzero(erow == 12) >= 14 else min(
                        3, int(np.count_nonzero(erow == 1)) // 4
                    )
                    return {"e": (x, y), "f": (x, fy),
                            "ep": ep, "fp": offset // 2}

    x, y = candidates[0]
    row = a[y, x:x + 16]
    ep = 4 if np.count_nonzero(row == 12) >= 14 else min(
        3, int(np.count_nonzero(row == 1)) // 4
    )
    return {"e": (x, y), "f": None, "ep": ep, "fp": 0}


def build_world(a, player, memory):
    y0, y1, cut = bounds(a)
    buttons, controls = buttons_from_frame(a, cut, y0, y1)
    px, py = player
    parity = (px % 2, py % 2)
    bridges = find_bridges(a, cut, parity)
    bridge_area = set()
    for hs, vs, _ in bridges:
        bridge_area.update(hs)
        bridge_area.update(vs)

    moving = find_moving(a, cut, buttons, parity)
    moving_area = set()
    if moving:
        for name in ("e", "f"):
            loc = moving.get(name)
            if loc:
                x, y = loc
                for xx in range(x, x + 16, 2):
                    moving_area.add((xx, y))

    base, walls, yellow, pink = set(), set(), {}, {}
    pads, keys, cells = [], [], set()
    goal = None

    for y in range(y0 + ((py - y0) % 2), min(y1, 61), 2):
        for x in range(px % 2, cut - 1, 2):
            pos = (x, y)
            cells.add(pos)
            tile = a[y:y + 2, x:x + 2]
            vals = set(int(v) for v in tile.flat)
            if 14 in vals:
                if pos not in bridge_area and pos not in moving_area:
                    base.add(pos)
                continue
            if 11 in vals:
                goal = pos
                base.add(pos)
            if vals == {6, 7}:
                pads.append(pos)
                base.add(pos)
            if (2 in vals or 1 in vals or 12 in vals) and pos not in moving_area:
                base.add(pos)
            if 0 in vals or 10 in vals:
                walls.add(pos)
            if 9 in vals:
                yellow[pos] = bool(np.all(tile == 9))
            if 15 in vals:
                if 2 in vals and np.count_nonzero(tile == 15) == 3:
                    keys.append(pos)
                    base.add(pos)
                elif not (2 in vals and np.count_nonzero(tile == 15) == 1):
                    pink[pos] = bool(np.all(tile == 15))

    if goal is None:
        goal = memory.get("goal")
    if goal is not None:
        memory["goal"] = goal
    if "d" in buttons:
        keys = []

    crane = None
    if controls:
        tool = find_tool(a, cut)
        cursor = find_cursor(a, cut)
        zeros = np.argwhere(a[:63, :cut] == 0)
        if tool is not None and cursor is not None and len(zeros) >= 20:
            zx = int(np.min(zeros[:, 1]))
            zy = int(np.max(zeros[:, 0]))
            ox, oy = zx - 1, zy - 5
            _, cx, cy, grabbed = cursor
            if (cx - ox) % 4 == 0 and (oy - cy) % 4 == 0:
                dx, up = (cx - ox) // 4, (oy - cy) // 4
                if 0 <= dx <= 3 and 0 <= up <= 3 and (dx == 0 or up in (0, 3)):
                    shape = set()
                    tx, ty = tool
                    for yy in range(0, 12, 2):
                        for xx in range(0, 6, 2):
                            if a[ty + yy, tx + xx] in (12, 14, 8):
                                shape.add((xx, yy))
                    if len(shape) >= 10:
                        for sx, sy in shape:
                            base.discard((tx + sx, ty + sy))
                        crane = {
                            "origin": (ox, oy), "start": (dx, up),
                            "grabbed": grabbed, "tool": tool,
                            "shape": shape, "controls": controls,
                        }
                        walls = {
                            pos for pos in walls
                            if not np.any(a[pos[1]:pos[1] + 2,
                                            pos[0]:pos[0] + 2] == 0)
                        }

    mode = 0
    if moving:
        mode |= moving["ep"] << 3
        mode |= moving["fp"] << 6

    return {
        "y0": y0, "y1": y1, "cut": cut, "cells": cells,
        "base": base, "walls": walls, "yellow": yellow, "pink": pink,
        "bridges": bridges, "bridge_area": bridge_area,
        "moving": moving, "moving_area": moving_area,
        "pads": pads, "keys": keys, "goal": goal,
        "buttons": buttons, "crane": crane, "initial_mode": mode,
    }


def update_old_tile(a, w, old, current, mode):
    if old is None or old == current or old not in w["cells"]:
        return
    if old in w["bridge_area"] or old in w["moving_area"]:
        return
    crane = w["crane"]
    if crane:
        tx, ty = crane["tool"]
        if (old[0] - tx, old[1] - ty) in crane["shape"]:
            return
    x, y = old
    tile = a[y:y + 2, x:x + 2]
    vals = set(int(v) for v in tile.flat)
    if 14 in vals:
        return
    if 2 in vals or 1 in vals or 12 in vals or 11 in vals or vals == {6, 7}:
        w["base"].add(old)
    else:
        w["base"].discard(old)
    if (0 in vals and not crane) or 10 in vals:
        w["walls"].add(old)
    else:
        w["walls"].discard(old)
    if 9 in vals:
        w["yellow"][old] = bool(np.all(tile == 9)) ^ bool(mode & 2)
    if 15 in vals and not (2 in vals and np.count_nonzero(tile == 15) == 3):
        w["pink"][old] = bool(np.all(tile == 15)) ^ bool(mode & 4)
    if vals == {6, 7} and old not in w["pads"]:
        w["pads"].append(old)


def tool_support(crane, pos, crane_pos, grabbed):
    if not crane:
        return False
    if grabbed:
        ox, oy = crane["origin"]
        dx, up = crane_pos
        tx = ox + dx * 4 + 6
        ty = oy - up * 4 - 2
    else:
        tx, ty = crane["tool"]
    return (pos[0] - tx, pos[1] - ty) in crane["shape"]


def crane_collision(crane, pos, crane_pos, grabbed):
    if not crane:
        return False
    ox, oy = crane["origin"]
    dx, up = crane_pos
    x0, y0 = ox + dx * 4, oy - up * 4
    if grabbed:
        rows = (
            "",
            "...#####",
            ".####..#",
            "####",
            "####",
            ".####..#",
            "...#####",
            "",
        )
    else:
        rows = (
            "...#####",
            "...#...#",
            ".####",
            "####",
            "####",
            ".####",
            "...#...#",
            "...#####",
        )
    x, y = pos
    for yy in range(y, y + 2):
        ry = yy - y0
        if not 0 <= ry < 8:
            continue
        row = rows[ry]
        for xx in range(x, x + 2):
            rx = xx - x0
            if 0 <= rx < len(row) and row[rx] == "#":
                return True
    return False


def moving_support(w, pos, mode):
    moving = w["moving"]
    if not moving:
        return False
    x, y = pos
    ex, ey = moving["e"]
    if y == ey and ex <= x < ex + 16:
        phase = (mode >> 3) & 7
        width = phase if phase <= 4 else 8 - phase
        if phase == 4:
            return True
        if width > 0 and (x < ex + 2 * width or x >= ex + 16 - 2 * width):
            return True
    if moving["f"]:
        fx, fy = moving["f"]
        phase = (mode >> 6) & 7
        if y == fy and fx + 2 * phase <= x < fx + 2 * phase + 6:
            return True
    return False


def safe(w, pos, mode, crane_pos, grabbed, check_crane=True):
    if pos not in w["cells"] or pos in w["walls"]:
        return False
    if check_crane and crane_collision(w["crane"], pos, crane_pos, grabbed):
        return False

    support = (
        pos in w["base"] or
        moving_support(w, pos, mode) or
        tool_support(w["crane"], pos, crane_pos, grabbed)
    )
    for hs, vs, initially_horizontal in w["bridges"]:
        horizontal = initially_horizontal ^ bool(mode & 1)
        if pos in (hs if horizontal else vs):
            support = True

    if pos in w["yellow"]:
        initial_solid = w["yellow"][pos]
        solid = initial_solid ^ bool(mode & 2)
        if solid:
            support = True
        elif not initial_solid:
            return False

    if pos in w["pink"]:
        initial_solid = w["pink"][pos]
        solid = initial_solid ^ bool(mode & 4)
        if solid:
            support = True
        elif not initial_solid:
            return False

    return support


def search(w, initial, blocked, visits):
    goal = w["goal"]
    keys = set(w["keys"])
    buttons = w["buttons"]
    crane = w["crane"]
    moving = w["moving"]
    queue = deque([initial])
    parent = {initial: None}
    best_state = None
    best_score = float("inf")

    def reconstruct(state):
        path = []
        while parent[state] is not None:
            prev, action = parent[state]
            path.append((prev, action, state))
            state = prev
        path.reverse()
        return path

    while queue and len(parent) < 160000:
        state = queue.popleft()
        pos, mode, cp, grabbed = state
        if state != initial:
            if pos == goal or pos in keys:
                return reconstruct(state), True
            if goal is not None:
                score = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
                score += 6 * visits.get(pos, 0)
                if score < best_score:
                    best_score, best_state = score, state

        def add(new_state, action):
            key = (state, action[0], action[1], action[2])
            if key in blocked or new_state in parent:
                return
            parent[new_state] = (state, action)
            queue.append(new_state)

        for dx, dy, name in (
            (0, -2, "ACTION1"), (0, 2, "ACTION2"),
            (-2, 0, "ACTION3"), (2, 0, "ACTION4")
        ):
            dest = (pos[0] + dx, pos[1] + dy)
            if safe(w, dest, mode, cp, grabbed):
                add((dest, mode, cp, grabbed), (name, None, None))

        for name, bit in (("a", 1), ("b", 2), ("d", 4)):
            if name in buttons:
                new_mode = mode ^ bit
                # A click checks support, but does not test sprite collisions.
                if safe(w, pos, new_mode, cp, grabbed, False):
                    row, col = buttons[name]
                    add((pos, new_mode, cp, grabbed), ("ACTION6", row, col))

        if "c" in buttons and pos in w["pads"]:
            for dest in w["pads"]:
                if dest != pos and safe(w, dest, mode, cp, grabbed, False):
                    row, col = buttons["c"]
                    add((dest, mode, cp, grabbed), ("ACTION6", row, col))

        if moving:
            for name in ("e", "f"):
                loc = moving.get(name)
                if not loc or name not in buttons:
                    continue
                x, y = loc
                if abs(pos[1] - y) > 12 or not x - 10 <= pos[0] <= x + 24:
                    continue
                if name == "e":
                    phase = (mode >> 3) & 7
                    new_mode = (mode & ~(7 << 3)) | (((phase + 1) % 8) << 3)
                else:
                    phase = (mode >> 6) & 7
                    new_mode = (mode & ~(7 << 6)) | (((phase + 1) % 6) << 6)
                if safe(w, pos, new_mode, cp, grabbed, False):
                    row, col = buttons[name]
                    add((pos, new_mode, cp, grabbed), ("ACTION6", row, col))

        if crane:
            dx, up = cp
            controls = crane["controls"]
            for ddx, dup, name in (
                (-1, 0, "left"), (1, 0, "right"),
                (0, 1, "up"), (0, -1, "down")
            ):
                ndx, nup = dx + ddx, up + dup
                if not (0 <= ndx <= 3 and 0 <= nup <= 3):
                    continue
                if not (ndx == 0 or nup in (0, 3)):
                    continue
                if ddx and up not in (0, 3):
                    continue
                if dup and dx != 0:
                    continue
                new_cp = (ndx, nup)
                if safe(w, pos, mode, new_cp, grabbed, False):
                    row, col = controls[name]
                    add((pos, mode, new_cp, grabbed), ("ACTION6", row, col))
            if not grabbed:
                ox, oy = crane["origin"]
                tx, ty = crane["tool"]
                anchor = (ox + dx * 4 + 9, oy - up * 4 + 4)
                if anchor == (tx + 3, ty + 6):
                    if safe(w, pos, mode, cp, 1, False):
                        row, col = controls["grab"]
                        add((pos, mode, cp, 1), ("ACTION6", row, col))

    if best_state is not None:
        return reconstruct(best_state), False
    return [], False


def signature(a, cut):
    part = a[:63, :cut].astype(np.int64)
    yy, xx = np.indices(part.shape)
    weights = 1 + xx * 37 + yy * 113 + xx * yy * 7
    return int(np.sum((part + 1) * weights) % 1000000007)


def act(obs, available, memory):
    a = np.asarray(obs)
    player = find_player(a)
    if player is None:
        return ("ACTION1", None, None)

    if "world" not in memory:
        w = build_world(a, player, memory)
        memory["world"] = w
        memory["mode"] = w["initial_mode"]
        crane = w["crane"]
        memory["crane_pos"] = crane["start"] if crane else (0, 0)
        memory["grabbed"] = crane["grabbed"] if crane else 0
        memory["visits"] = {}
        memory["blocked"] = set()
        memory["last_pos"] = None
        memory["pending"] = None
        memory["plan"] = []

    w = memory["world"]
    blocked = memory["blocked"]
    pending = memory.get("pending")
    if pending:
        old_state, action, predicted, old_sig = pending
        name, row, col = action
        failed = player != predicted[0]
        if name == "ACTION6" and not failed:
            if signature(a, w["cut"]) == old_sig:
                failed = True
        if failed:
            blocked.add((old_state, name, row, col))
            memory["mode"] = old_state[1]
            memory["crane_pos"] = old_state[2]
            memory["grabbed"] = old_state[3]
            memory["plan"] = []
        else:
            memory["mode"] = predicted[1]
            memory["crane_pos"] = predicted[2]
            memory["grabbed"] = predicted[3]
        memory["pending"] = None

    mode = memory["mode"]
    update_old_tile(a, w, memory.get("last_pos"), player, mode)
    buttons, controls = buttons_from_frame(a, w["cut"], w["y0"], w["y1"])
    w["buttons"].update(buttons)
    if "d" in buttons and w["keys"]:
        w["keys"] = []
        memory["plan"] = []
    if w["crane"] and controls:
        w["crane"]["controls"] = controls

    visits = memory["visits"]
    visits[player] = visits.get(player, 0) + 1
    state = (player, memory["mode"], memory["crane_pos"], memory["grabbed"])

    plan = memory.get("plan", [])
    if plan and plan[0][0] != state:
        plan = []
    if not plan:
        plan, _ = search(w, state, blocked, visits)
    if not plan:
        counts = memory.setdefault("fallback_counts", {})
        for name in ("e", "f", "a", "b", "c", "d"):
            if name in buttons and counts.get(name, 0) < 2:
                counts[name] = counts.get(name, 0) + 1
                row, col = buttons[name]
                memory.pop("world", None)
                return ("ACTION6", int(row), int(col))
        return ("ACTION1", None, None)

    old_state, action, predicted = plan.pop(0)
    memory["plan"] = plan
    name, row, col = action
    if name not in available:
        memory["plan"] = []
        return ("ACTION1", None, None)

    memory["last_pos"] = player
    memory["pending"] = (old_state, action, predicted, signature(a, w["cut"]))
    if name == "ACTION6":
        return (name, int(row), int(col))
    return (name, None, None)

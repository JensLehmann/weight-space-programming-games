import numpy as np
from collections import deque
import heapq

_PATTERNS = {
    "fire": (1 << 1) | (1 << 4) | (1 << 7),
    "size": (1 << 1) | (1 << 3) | (1 << 5) | (1 << 7),
    "tele": (1 << 0) | (1 << 1) | (1 << 4),
}
_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _spell_path(start, spell, spells):
    target = _PATTERNS[spell]
    forbidden = {_PATTERNS[s] for s in spells if s != spell}
    q = deque([start])
    prev = {start: None}
    if start == target:
        return []
    while q:
        m = q.popleft()
        order = sorted(
            range(9),
            key=lambda i: (
                0 if (target & (1 << i)) and not (m & (1 << i))
                else 1 if (m & (1 << i)) and not (target & (1 << i))
                else 2,
                i,
            ),
        )
        for i in order:
            nm = m ^ (1 << i)
            if nm in forbidden or nm in prev:
                continue
            prev[nm] = (m, i)
            if nm == target:
                path = []
                while prev[nm] is not None:
                    nm, bit = prev[nm]
                    path.append(bit)
                return path[::-1]
            q.append(nm)
    return None


def _square_hits(board, n, outside=True):
    p = np.pad(board.astype(np.int32), ((1, 0), (1, 0)))
    p = p.cumsum(0).cumsum(1)
    sums = p[n:, n:] - p[:-n, n:] - p[n:, :-n] + p[:-n, :-n]
    out = np.full((64, 64), outside, dtype=bool)
    out[:65 - n, :65 - n] = sums > 0
    return out


def _rays(code):
    out = np.zeros((4, 64, 64), dtype=np.int16)
    last = np.zeros(64, dtype=np.int16)
    for y in range(64):
        out[0, y] = last
        last = np.where(code[y] != 0, code[y], last)
    last.fill(0)
    for y in range(63, -1, -1):
        out[1, y] = last
        last = np.where(code[y] != 0, code[y], last)
    last.fill(0)
    for x in range(64):
        out[2, :, x] = last
        last = np.where(code[:, x] != 0, code[:, x], last)
    last.fill(0)
    for x in range(63, -1, -1):
        out[3, :, x] = last
        last = np.where(code[:, x] != 0, code[:, x], last)
    return out


def _recognize_actors(a, ui, memory):
    goal_base = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ], dtype=np.int16)
    goal = np.zeros((64, 64), dtype=bool)
    points = [(int(y), int(x))
              for y, x in zip(*np.where(((a == 9) | (a == 10)) & ~ui))]
    for k in range(4):
        t = np.rot90(goal_base, k)
        h, w = t.shape
        for y, x in points:
            if y + h <= 64 and x + w <= 64:
                if np.array_equal(a[y:y + h, x:x + w], t):
                    goal[y:y + h, x:x + w] = True
    if goal.any():
        memory["goal"] = goal.copy()
    else:
        goal = memory.get("goal", goal).copy()

    base = np.array([[10, 10], [9, 9]], dtype=np.int16)
    candidates = []
    # A large player contains small-player-looking patches, so test large first.
    for size in (2, 1):
        n = 2 * size
        templates = [
            np.repeat(np.repeat(np.rot90(base, k), size, 0), size, 1)
            for k in range(4)
        ]
        for y, x in points:
            if y + n > 64 or x + n > 64:
                continue
            if goal[y:y + n, x:x + n].any():
                continue
            patch = a[y:y + n, x:x + n]
            for k, t in enumerate(templates):
                if np.array_equal(patch, t):
                    # np.rot90 is anticlockwise: down, right, up, left.
                    facing = (1, 3, 0, 2)[k]
                    candidates.append((x, y, size, facing))
                    break
        if candidates:
            break

    # Growth can overlap an exit, which renders above the player.
    if not candidates:
        predictions = []
        for key in ("final_cast", "previous_move"):
            data = memory.get(key)
            if data is not None:
                predictions.append(data.get("predicted"))
        predictions.append(memory.get("player"))
        for p in predictions:
            if p is None:
                continue
            x, y, size = p
            n = size * 2
            if not (0 <= x <= 64 - n and 0 <= y <= 64 - n):
                continue
            g = goal[y:y + n, x:x + n]
            if not g.any():
                continue
            visible = ~g
            patch = a[y:y + n, x:x + n]
            for k in range(4):
                t = np.repeat(np.repeat(np.rot90(base, k), size, 0), size, 1)
                if (not visible.any()
                        or np.array_equal(patch[visible], t[visible])):
                    candidates.append((x, y, size, (1, 3, 0, 2)[k]))
                    break
            if candidates:
                break

    if not candidates:
        return None, goal
    old = memory.get("player")
    if old:
        candidates.sort(key=lambda p: abs(p[0] - old[0]) + abs(p[1] - old[1]))
    return candidates[0], goal


def _read(a, memory):
    ui = np.zeros((64, 64), dtype=bool)
    ui[47:64, 22:39] = True
    ui[:, 62:64] = True
    spells = set()
    # Spell panels have a complete 10x10 green border.
    for y, x in zip(*np.where(a == 3)):
        y, x = int(y), int(x)
        if y > 54 or x > 54 or a[y + 9, x + 9] != 3:
            continue
        if (np.all(a[y, x:x + 10] == 3)
                and np.all(a[y + 9, x:x + 10] == 3)
                and np.all(a[y:y + 10, x] == 3)
                and np.all(a[y:y + 10, x + 9] == 3)):
            patch = a[y + 1:y + 9, x + 1:x + 9]
            ui[y:y + 10, x:x + 10] = True
            for color, spell in ((15, "size"), (11, "tele"), (6, "fire")):
                if np.any(patch == color):
                    spells.add(spell)
    memory.setdefault("spells", set()).update(spells)
    spells = set(memory["spells"])

    actor, goal = _recognize_actors(a, ui, memory)
    if actor is None:
        return None
    x, y, size, facing = actor

    selected = 0
    for i in range(9):
        if int(a[50 + 5 * (i // 3), 25 + 5 * (i % 3)]) not in (0, 2):
            selected |= 1 << i

    big = memory.setdefault("big", [])
    small = memory.setdefault("small", [])

    for cy, cx in zip(*np.where((a == 15) & ~ui)):
        cy, cx = int(cy), int(cx)
        if cy < 63 and cx < 63 and np.all(a[cy:cy + 2, cx:cx + 2] == 15):
            if (cx, cy) not in small:
                small.append((cx, cy))

    # Large highlights reveal covered teleport targets as well.
    for color, targets, offset in ((15, small, 2), (11, big, 1)):
        for cy, cx in zip(*np.where((a == color) & ~ui)):
            cy, cx = int(cy), int(cx)
            if cy + 5 >= 64 or cx + 5 >= 64:
                continue
            if (a[cy, cx + 5] == color and a[cy + 5, cx] == color
                    and a[cy + 5, cx + 5] == color):
                p = (cx + offset, cy + offset)
                if p not in targets:
                    targets.append(p)

    for cy, cx in zip(*np.where((a == 11) & ~ui)):
        cy, cx = int(cy), int(cx)
        if cy + 3 >= 64 or cx + 3 >= 64:
            continue
        if (a[cy, cx + 3] == 11 and a[cy + 3, cx] == 11
                and a[cy + 3, cx + 3] == 11):
            if (cx + 1, cy + 1) not in small and (cx, cy) not in big:
                big.append((cx, cy))

    pads = memory.setdefault("pads", set())
    for sx, sy in small:
        px, py = sx - 1, sy - 1
        if 0 <= px <= 60 and 0 <= py <= 60:
            corners = ((py, px), (py, px + 3),
                       (py + 3, px), (py + 3, px + 3))
            if all(a[yy, xx] == 11 for yy, xx in corners):
                pads.update(corners)

    bi = memory.get("bi", 0) % max(1, len(big))
    si = memory.get("si", 0) % max(1, len(small))
    if size == 2:
        for i, (tx, ty) in enumerate(big):
            corners = ((ty - 1, tx - 1), (ty - 1, tx + 4),
                       (ty + 4, tx - 1), (ty + 4, tx + 4))
            if all(0 <= yy < 64 and 0 <= xx < 64 and a[yy, xx] == 11
                   for yy, xx in corners):
                bi = i
                break
    memory["bi"], memory["si"] = bi, si

    switches = []
    switch_pixels = np.zeros((64, 64), dtype=bool)
    for cy, cx in zip(*np.where((a == 6) & ~ui)):
        cy, cx = int(cy), int(cx)
        if cy > 60 or cx > 60:
            continue
        p = a[cy:cy + 4, cx:cx + 4]
        if (np.all(p[0] == 6) and np.all(p[3] == 6)
                and np.all(p[:, 0] == 6) and np.all(p[:, 3] == 6)):
            color = int(p[1, 1])
            if color in (12, 13) and np.all(p[1:3, 1:3] == color):
                switches.append((cx, cy, 0 if color == 13 else 1))
                switch_pixels[cy:cy + 4, cx:cx + 4] = True

    gates = [(a == c) & ~ui & ~switch_pixels for c in (13, 12)]
    walls = (a == 5) & ~ui
    if "walls" not in memory:
        memory["walls"] = walls.copy()
    else:
        memory["walls"] |= walls

    solid = ~np.isin(a, (2, 9, 10, 11, 15, 14))
    solid |= memory["walls"]
    solid[ui] = True
    solid[gates[0] | gates[1] | switch_pixels] = False
    for yy, xx in pads:
        solid[yy, xx] = True

    growth = solid.copy()
    growth[((a == 4) | (a == 6)) & ~ui] = False
    for yy, xx in pads:
        growth[yy, xx] = False

    signature = (
        tuple(switches),
        np.packbits(gates[0]).tobytes(),
        np.packbits(gates[1]).tobytes(),
    )
    return {
        "player": (x, y, size), "facing": facing,
        "selected": selected, "spells": spells, "goal": goal,
        "solid": solid, "growth": growth,
        "gates": gates, "switches": switches,
        "big": big, "small": small, "bi": bi, "si": si,
        "signature": signature,
    }


def _search(env, memory, movement, clickable):
    spells = env["spells"] if clickable else set()
    first_paths = {s: _spell_path(env["selected"], s, spells) for s in spells}
    usual_paths = {s: _spell_path(0, s, spells) for s in spells}
    x, y, size = env["player"]
    direction = memory.get("direction", 4)
    if direction not in range(4):
        direction = 4
    root = (x, y, size, direction, 0, env["bi"], env["si"],
            int(bool(env["selected"])))
    distances = {root: 0}
    parents = {}
    queue = [(0, root)]
    cached_maps = {}
    cached_growth = {}
    overrides = memory.setdefault("overrides", {})
    signature = env["signature"]
    switches = env["switches"]
    goal_maps = {
        1: _square_hits(env["goal"], 2, False),
        2: _square_hits(env["goal"], 4, False),
    }
    gy, gx = np.where(env["goal"])
    if not len(gx):
        return None
    center = (float(gx.mean()), float(gy.mean()))

    def maps(mask):
        if mask in cached_maps:
            return cached_maps[mask]
        solid = env["solid"].copy()
        growth = env["growth"].copy()
        opened = [False, False]
        for i, (_, _, kind) in enumerate(switches):
            if mask & (1 << i):
                opened[kind] = True
        for kind in range(2):
            if not opened[kind]:
                solid |= env["gates"][kind]
                growth |= env["gates"][kind]
        code = np.where(growth, -1, 0).astype(np.int16)
        for i, (sx, sy, _) in enumerate(switches):
            if not mask & (1 << i):
                solid[sy:sy + 4, sx:sx + 4] = True
                code[sy:sy + 4, sx:sx + 4] = i + 1
        result = (_square_hits(solid, 2), _square_hits(solid, 4),
                  _square_hits(growth, 4), _rays(code))
        cached_maps[mask] = result
        return result

    def first_edge(state, final_action=None, final_state=None):
        if state == root:
            return final_action, final_state
        child = state
        while parents[child][0] != root:
            child = parents[child][0]
        return parents[child][1], child

    best = root
    best_rank = (abs(x - center[0]) + abs(y - center[1]), 0, 0)
    expanded = 0
    while queue and expanded < 90000:
        cost, state = heapq.heappop(queue)
        if cost != distances.get(state):
            continue
        expanded += 1
        if cost > 110:
            continue
        px, py, s, d, mask, bi, si, partial = state
        rank = (abs(px - center[0]) + abs(py - center[1]),
                -bin(mask).count("1"), cost)
        if rank < best_rank:
            best_rank, best = rank, state
        c2, c4, growth4, rays = maps(mask)

        def add(ns, extra, action):
            nc = cost + extra
            if nc < distances.get(ns, 10**9):
                distances[ns] = nc
                parents[ns] = (state, action)
                heapq.heappush(queue, (nc, ns))

        for action in movement:
            nd = action - 1
            dx, dy = _DELTAS[nd]
            nx, ny = px, py
            won = False
            override = (overrides.get((signature, px, py, s, action))
                        if mask == 0 else None)
            if override is not None:
                nx, ny = override[:2]
            else:
                steps = (4, 2) if s == 2 else (2,)
                for step in steps:
                    tx, ty = px + dx * step, py + dy * step
                    if not (0 <= tx < 64 and 0 <= ty < 64):
                        continue
                    hits_goal = goal_maps[s][ty, tx]
                    blocked = (c4 if s == 2 else c2)[ty, tx] or hits_goal
                    if not blocked:
                        nx, ny = tx, ty
                        break
                    if step == steps[-1] and hits_goal:
                        won = True
            ns = (nx, ny, s, nd, mask, bi, si, partial)
            if won:
                return first_edge(state, action, ns)
            if ns != state:
                add(ns, 1, action)

        def spell_cost(spell):
            path = first_paths[spell] if partial else usual_paths[spell]
            return None if path is None else len(path) + 1

        if "size" in spells:
            sc = spell_cost("size")
            if sc is not None:
                override = (overrides.get((signature, px, py, s, 0))
                            if mask == 0 else None)
                if override is not None:
                    nx, ny, new_size = override
                elif s == 2:
                    nx, ny, new_size = px, py, 1
                else:
                    key = (mask, px, py)
                    if key not in cached_growth:
                        result = None
                        for ox in range(-2, 1):
                            for oy in range(-2, 1):
                                tx, ty = px + ox, py + oy
                                if (0 <= tx < 64 and 0 <= ty < 64
                                        and not growth4[ty, tx]):
                                    result = (tx, ty)
                                    break
                            if result is not None:
                                break
                        cached_growth[key] = result
                    result = cached_growth[key]
                    if result is None:
                        nx, ny, new_size = px, py, 1
                    else:
                        nx, ny = result
                        new_size = 2
                if (nx, ny, new_size) != (px, py, s):
                    add((nx, ny, new_size, d, mask, bi, si, 0), sc, "size")

        if "tele" in spells:
            targets = env["big"] if s == 2 else env["small"]
            sc = spell_cost("tele")
            if targets and sc is not None:
                index = (bi if s == 2 else si) % len(targets)
                nx, ny = targets[index]
                nbi, nsi = bi, si
                if s == 2:
                    nbi = (index + 1) % len(targets)
                else:
                    nsi = (index + 1) % len(targets)
                ns = (nx, ny, s, d, mask, nbi, nsi, 0)
                if ns != state:
                    add(ns, sc, "tele")

        if "fire" in spells and d < 4:
            n = 2 * s
            rx = px + (n - 1 if d == 3 else 0)
            ry = py + (n - 1 if d == 1 else 0)
            if 0 <= rx < 64 and 0 <= ry < 64:
                hit = int(rays[d, ry, rx])
                if 1 <= hit <= len(switches) and not mask & (1 << (hit - 1)):
                    sc = spell_cost("fire")
                    if sc is not None:
                        add((px, py, s, d, mask | (1 << (hit - 1)),
                             bi, si, 0), sc, "fire")

    if best != root:
        return first_edge(best)
    return None


def act(obs, available, memory):
    a = np.asarray(obs, dtype=np.int16)
    available = set(available)
    env = _read(a, memory)

    def fallback(reset=False):
        if reset and "RESET" in available:
            memory.clear()
            return "RESET", None, None
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if name in available:
                return name, None, None
        if "ACTION6" in available:
            return "ACTION6", 50, 25
        return "RESET", None, None

    if env is None:
        return fallback(True)

    player = env["player"]
    overrides = memory.setdefault("overrides", {})
    previous = memory.pop("previous_move", None)
    if previous is not None:
        action = previous["action"]
        if env["facing"] == action - 1:
            memory["direction"] = action - 1
            if player != previous["predicted"]:
                ox, oy, os = previous["source"]
                overrides[(previous["signature"], ox, oy, os, action)] = player
        else:
            memory["direction"] = 4

    cast = memory.pop("final_cast", None)
    if cast is not None:
        if env["selected"] != 0:
            # A tutorial can consume a click without entering its cell.
            memory["want"] = cast
        else:
            memory.pop("want", None)
            if cast["spell"] == "size" and player != cast["predicted"]:
                ox, oy, os = cast["source"]
                overrides[(cast["signature"], ox, oy, os, 0)] = player
            elif cast["spell"] == "tele":
                old_size = cast["source"][2]
                targets = env["big"] if old_size == 2 else env["small"]
                position = player[:2]
                if position != cast["source"][:2]:
                    if position not in targets:
                        targets.append(position)
                    index = (targets.index(position) + 1) % len(targets)
                    key = "bi" if old_size == 2 else "si"
                    memory[key] = index
                    env[key] = index

    memory["player"] = player
    clickable = "ACTION6" in available
    want = memory.get("want")
    if want is not None and want["spell"] not in env["spells"]:
        memory.pop("want", None)
        want = None

    if want is None:
        movement = [i for i in range(1, 5) if "ACTION%d" % i in available]
        plan = _search(env, memory, movement, clickable)
        if plan is None:
            # If reconstruction is temporarily incomplete, use a transformation.
            choices = [s for s in ("tele", "size", "fire") if s in env["spells"]]
            if not clickable or not choices:
                return fallback()
            count = memory.get("fallback_count", 0)
            memory["fallback_count"] = count + 1
            spell = choices[count % len(choices)]
            want = {
                "spell": spell, "source": player, "predicted": player,
                "signature": env["signature"],
            }
        else:
            action, next_state = plan
            prediction = next_state[:3]
            if isinstance(action, int):
                memory["previous_move"] = {
                    "action": action, "source": player,
                    "predicted": prediction, "signature": env["signature"],
                }
                return "ACTION%d" % action, None, None
            want = {
                "spell": action, "source": player,
                "predicted": prediction, "signature": env["signature"],
            }
        memory["want"] = want

    if not clickable:
        memory.pop("want", None)
        return fallback()

    path = _spell_path(env["selected"], want["spell"], env["spells"])
    if not path:
        memory.pop("want", None)
        return fallback()
    bit = path[0]
    if len(path) == 1:
        memory["final_cast"] = dict(want)
        memory.pop("want", None)
    return "ACTION6", 50 + 5 * (bit // 3), 25 + 5 * (bit % 3)

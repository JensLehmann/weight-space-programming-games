import math
import heapq
import time
from collections import deque
import numpy as np

# Spell patterns, in row-major order.
_PATTERNS = {"tele": 19, "size": 170, "fire": 146}
_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for yy, xx in np.argwhere(mask):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(x, y)]
        pixels = []
        while stack:
            x, y = stack.pop()
            pixels.append((x, y))
            for dx, dy in _DIRS:
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        xs, ys = zip(*pixels)
        result.append((min(xs), min(ys),
                       max(xs) - min(xs) + 1, max(ys) - min(ys) + 1,
                       pixels))
    return result


def _rect_overlap(a, b):
    x, y, w, h = a
    X, Y, W, H = b
    return x < X + W and X < x + w and y < Y + H and Y < y + h


def _paint(a, rect, value):
    x, y, w, h = rect
    a[max(0, y):min(64, y + h), max(0, x):min(64, x + w)] = value


def _grid(obs):
    mask = 0
    for i in range(9):
        if int(obs[50 + 5 * (i // 3), 25 + 5 * (i % 3)]) in (6, 11, 14, 15):
            mask |= 1 << i
    return mask


def _cards(obs):
    ui = np.zeros((64, 64), dtype=bool)
    ui[47:64, 22:39] = True
    ui[:, 62:64] = True
    spells = set()
    for yy, xx in np.argwhere(obs == 3):
        x, y = int(xx), int(yy)
        if x > 54 or y > 54:
            continue
        a = obs[y:y + 10, x:x + 10]
        if (np.all(a[0] == 3) and np.all(a[-1] == 3) and
                np.all(a[:, 0] == 3) and np.all(a[:, -1] == 3)):
            ui[y:y + 10, x:x + 10] = True
            inside = a[1:9, 1:9]
            if np.any(inside == 6):
                spells.add("fire")
            if np.any(inside == 11):
                spells.add("tele")
            if np.any(inside == 15):
                spells.add("size")
    return ui, tuple(s for s in ("fire", "tele", "size") if s in spells)


def _locate(obs, old_goals=()):
    door = np.array([[9] * 5] +
                    [[9, 10, 10, 10, 10] for _ in range(4)] +
                    [[9] * 5], dtype=np.int8)
    goals = []
    candidates = np.argwhere(obs == 9)
    for k in range(4):
        t = np.rot90(door, -k)
        h, w = t.shape
        for yy, xx in candidates:
            x, y = int(xx), int(yy)
            if x + w <= 64 and y + h <= 64:
                if np.array_equal(obs[y:y + h, x:x + w], t):
                    r = (x, y, w, h)
                    if r not in goals:
                        goals.append(r)
    if not goals:
        goals = list(old_goals)
    excluded = np.zeros((64, 64), dtype=bool)
    for r in goals:
        _paint(excluded, r, True)
    points = np.argwhere(((obs == 9) | (obs == 10)) & ~excluded)
    base = np.array([[10, 10], [9, 9]], dtype=np.int8)
    rotation_direction = (1, 2, 0, 3)
    for scale in (2, 1):
        for k in range(4):
            t = np.rot90(base, -k)
            t = np.repeat(np.repeat(t, scale, axis=0), scale, axis=1)
            size = 2 * scale
            for yy, xx in points:
                x, y = int(xx), int(yy)
                if x + size > 64 or y + size > 64:
                    continue
                if excluded[y:y + size, x:x + size].any():
                    continue
                if np.array_equal(obs[y:y + size, x:x + size], t):
                    return (x, y, scale, rotation_direction[k]), goals
    return None, goals


def _corners(obs, color, distance, ui):
    found = []
    for yy, xx in np.argwhere((obs == color) & ~ui):
        x, y = int(xx), int(yy)
        d = distance
        if x + d < 64 and y + d < 64:
            if (obs[y, x + d] == color and
                    obs[y + d, x] == color and
                    obs[y + d, x + d] == color and
                    not ui[y + d, x + d]):
                found.append((x, y))
    return found


def _markers(obs, ui, player):
    small = []
    for x, y, w, h, pixels in _components((obs == 15) & ~ui):
        if w == h == 2 and len(pixels) == 4:
            small.append((x, y))
    for x, y in _corners(obs, 15, 5, ui):
        p = (x + 2, y + 2)
        if p not in small:
            small.append(p)

    large = []
    collider = []
    px, py, scale, _ = player
    for x, y in _corners(obs, 11, 3, ui):
        inside_small = (x + 1, y + 1) in small
        inside_player = scale == 1 and (px, py) == (x + 1, y + 1)
        if inside_small or inside_player:
            collider.extend(((x, y), (x + 3, y),
                             (x, y + 3), (x + 3, y + 3)))
            if inside_player and (px, py) not in small:
                small.append((px, py))
        elif (x, y) not in large:
            large.append((x, y))

    active = []
    for x, y in _corners(obs, 11, 5, ui):
        p = (x + 1, y + 1)
        active.append(p)
        if p not in large:
            large.append(p)
    large.sort(key=lambda p: (p[1], p[0]))
    small.sort(key=lambda p: (p[1], p[0]))
    return large, small, active, collider


def _box_or(a, size, outside=0):
    """OR of every size-by-size window, indexed by its upper-left pixel."""
    result = np.full((64, 64), outside, dtype=a.dtype)
    n = 65 - size
    inner = np.zeros((n, n), dtype=a.dtype)
    for dy in range(size):
        for dx in range(size):
            inner |= a[dy:dy + n, dx:dx + n]
    result[:n, :n] = inner
    return result.ravel()


def _background_under(obs, x, y):
    count = 0
    for dx, dy in _DIRS:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < 64 and 0 <= ny < 64) or obs[ny, nx] == 5:
            count += 1
    return count >= 2


def _build_world(obs, player, goals):
    ui, spells = _cards(obs)
    large, small, active, collider = _markers(obs, ui, player)
    targets = []
    for yy, xx in np.argwhere((obs == 6) & ~ui):
        x, y = int(xx), int(yy)
        if x > 60 or y > 60:
            continue
        a = obs[y:y + 4, x:x + 4]
        if (np.all(a[0] == 6) and np.all(a[-1] == 6) and
                np.all(a[:, 0] == 6) and np.all(a[:, -1] == 6)):
            for color in (13, 12):
                if np.all(a[1:3, 1:3] == color):
                    item = (x, y, 4, 4, color)
                    if item not in targets:
                        targets.append(item)

    target_area = np.zeros((64, 64), dtype=bool)
    for item in targets:
        _paint(target_area, item[:4], True)
    gates = {13: [], 12: []}
    for color in (13, 12):
        for x, y, w, h, pixels in _components(
                (obs == color) & ~ui & ~target_area):
            gates[color].append((x, y, w, h))

    bonuses = []
    for x, y, w, h, pixels in _components((obs == 14) & ~ui):
        if w == h == 2 and len(pixels) == 4:
            bonuses.append((x, y, w, h))

    static = obs != 2
    for yy, xx in np.argwhere(((obs == 11) | (obs == 15)) & ~ui):
        x, y = int(xx), int(yy)
        static[y, x] = _background_under(obs, x, y)
    for x, y in large:
        for dx, dy in ((0, 0), (3, 0), (0, 3), (3, 3)):
            if 0 <= x + dx < 64 and 0 <= y + dy < 64:
                static[y + dy, x + dx] = False
    for x, y in small:
        _paint(static, (x, y, 2, 2), False)
    for r in goals:
        _paint(static, r, False)
    for item in targets:
        _paint(static, item[:4], False)
    for color in gates:
        for r in gates[color]:
            _paint(static, r, False)
    for r in bonuses:
        _paint(static, r, False)
    px, py, scale, _ = player
    _paint(static, (px, py, 2 * scale, 2 * scale), False)
    static[ui] = True

    growth = static.copy()
    # Recognize the apertures that block walking but not fire or resizing.
    aperture = np.array([[1, 1, 0, 0, 1, 1],
                         [0, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 1, 0],
                         [1, 1, 0, 0, 1, 1]], dtype=bool)
    fours = np.argwhere((obs == 4) & ~ui)
    for k in range(4):
        t = np.rot90(aperture, k)
        h, w = t.shape
        for yy, xx in fours:
            x, y = int(xx), int(yy)
            if x + w <= 64 and y + h <= 64:
                a = obs[y:y + h, x:x + w]
                if np.all(a[t] == 4):
                    for dy, dx in np.argwhere(t):
                        X, Y = x + int(dx), y + int(dy)
                        growth[Y, X] = _background_under(obs, X, Y)
    for x, y in collider:
        static[y, x] = True
        growth[y, x] = _background_under(obs, x, y)

    n = len(targets)
    object_map = np.zeros((64, 64), dtype=np.int64)
    gate_map = np.zeros((64, 64), dtype=np.int64)
    alive = 0
    clear_bits = []
    for i, item in enumerate(targets):
        bit = 1 << i
        _paint(object_map, item[:4], bit)
        alive |= bit
        group = n + (0 if item[4] == 13 else 1)
        clear_bits.append(bit | (1 << group))
    for j, color in enumerate((13, 12)):
        bit = 1 << (n + j)
        for r in gates[color]:
            x, y, w, h = r
            object_map[y:y + h, x:x + w] |= bit
            gate_map[y:y + h, x:x + w] |= bit
            alive |= bit

    goal_map = np.zeros((64, 64), dtype=bool)
    for r in goals:
        _paint(goal_map, r, True)
    bonus_map = np.zeros((64, 64), dtype=np.int64)
    for i, r in enumerate(bonuses):
        x, y, w, h = r
        bonus_map[y:y + h, x:x + w] |= 1 << i

    return {
        "ui": ui, "spells": spells, "goals": goals,
        "large": large, "small": small,
        "targets": targets, "gates": gates, "bonuses": bonuses,
        "alive": alive, "clear": clear_bits,
        "static": [_box_or(static, s, True) for s in (2, 4)],
        "objects": [_box_or(object_map, s) for s in (2, 4)],
        "goalhit": [_box_or(goal_map, s) for s in (2, 4)],
        "grow": _box_or(growth, 4, True),
        "growgate": _box_or(gate_map, 4),
        "raywall": growth.ravel(),
        "rayobject": object_map.ravel(),
        "bonus": [_box_or(bonus_map, s) for s in (2, 4)],
        "initial_large": large.index(active[0]) if active and active[0] in large else 0,
        "move_cache": {}, "grow_cache": {}, "ray_cache": {},
    }


def _observe_dynamic(obs, world, player, alive, bonus):
    px, py, scale, _ = player
    pr = (px, py, 2 * scale, 2 * scale)
    for i, item in enumerate(world["targets"]):
        x, y, w, h, color = item
        visible = np.any(obs[y:y + h, x:x + w] == 6)
        if visible:
            alive |= 1 << i
        elif not _rect_overlap(item[:4], pr):
            alive &= ~(1 << i)
    n = len(world["targets"])
    for j, color in enumerate((13, 12)):
        rectangles = world["gates"][color]
        if not rectangles:
            continue
        visible = any(np.any(obs[y:y + h, x:x + w] == color)
                      for x, y, w, h in rectangles)
        if visible:
            alive |= 1 << (n + j)
        elif not any(_rect_overlap(r, pr) for r in rectangles):
            alive &= ~(1 << (n + j))
    for i, r in enumerate(world["bonuses"]):
        x, y, w, h = r
        if np.any(obs[y:y + h, x:x + w] == 14):
            bonus &= ~(1 << i)
        elif not _rect_overlap(r, pr):
            bonus |= 1 << i
    return alive, bonus


def _spell_path(start, spell, spells):
    target = _PATTERNS[spell]
    forbidden = {_PATTERNS[s] for s in spells if s != spell}
    q = deque([start])
    previous = {start: None}
    last_bit = {}
    while q:
        mask = q.popleft()
        if mask == target and mask != start:
            path = []
            while previous[mask] is not None:
                path.append(last_bit[mask])
                mask = previous[mask]
            return path[::-1]
        # Prefer changes that approach the requested pattern.
        bits = sorted(range(9), key=lambda i: not ((mask ^ target) & (1 << i)))
        for i in bits:
            nxt = mask ^ (1 << i)
            if nxt in forbidden or nxt in previous:
                continue
            previous[nxt] = mask
            last_bit[nxt] = i
            q.append(nxt)
    return None


def _moves(world, p, big, alive):
    key = (p, big, alive)
    cache = world["move_cache"]
    if key in cache:
        return cache[key]
    x, y = p % 64, p // 64
    size = 4 if big else 2
    result = []
    for dx, dy in _DIRS:
        outcome = (p, False)
        for step in ((4, 2) if big else (2,)):
            nx, ny = x + dx * step, y + dy * step
            if not (0 <= nx <= 64 - size and 0 <= ny <= 64 - size):
                hit, goal, np_ = True, False, p
            else:
                np_ = ny * 64 + nx
                goal = bool(world["goalhit"][big][np_])
                hit = (bool(world["static"][big][np_]) or goal or
                       bool(int(world["objects"][big][np_]) & alive))
            if not hit:
                outcome = (np_, False)
                break
            outcome = (p, goal)
        result.append(outcome)
    cache[key] = result
    return result


def _grow(world, p, alive):
    key = (p, alive)
    cache = world["grow_cache"]
    if key in cache:
        return cache[key]
    x, y = p % 64, p // 64
    answer = None
    for dx in range(-2, 1):
        for dy in range(-2, 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx <= 60 and 0 <= ny <= 60:
                q = ny * 64 + nx
                if (not world["grow"][q] and
                        not (int(world["growgate"][q]) & alive)):
                    answer = q
                    break
        if answer is not None:
            break
    cache[key] = answer
    return answer


def _fire_target(world, p, big, direction, alive):
    key = (p, big, direction, alive)
    cache = world["ray_cache"]
    if key in cache:
        return cache[key]
    x, y = p % 64, p // 64
    dx, dy = _DIRS[direction]
    size = 4 if big else 2
    if dx > 0:
        x += size - 1
    if dy > 0:
        y += size - 1
    answer = -1
    while True:
        x, y = x + dx, y + dy
        if not (0 <= x < 64 and 0 <= y < 64):
            break
        q = y * 64 + x
        if world["raywall"][q]:
            break
        hit = int(world["rayobject"][q]) & alive
        if hit:
            i = (hit & -hit).bit_length() - 1
            if i < len(world["targets"]):
                answer = i
            break
    cache[key] = answer
    return answer


def _search(world, initial, grid, remaining, deadline):
    spells = world["spells"]
    paths = {}
    for g in (0, 1):
        for spell in spells:
            paths[g, spell] = _spell_path(grid if g else 0, spell, spells)

    start_bonus_count = initial[7].bit_count()
    goals = world["goals"]
    heuristic_cache = {}

    def walking_h(p):
        x, y = p % 64, p // 64
        best = 100
        for gx, gy, gw, gh in goals:
            dx = max(gx - (x + 3), x - (gx + gw - 1), 0)
            dy = max(gy - (y + 3), y - (gy + gh - 1), 0)
            best = min(best, max(0, (dx + dy + 3) // 4 - 1))
        return best

    tele_bound = 100
    if "tele" in spells and world["large"] + world["small"]:
        costs = [len(paths[g, "tele"]) + 1 for g in (0, 1)
                 if paths[g, "tele"]]
        if costs:
            tele_bound = min(costs) + min(
                walking_h(y * 64 + x)
                for x, y in world["large"] + world["small"])

    def heuristic(p):
        if p not in heuristic_cache:
            heuristic_cache[p] = min(walking_h(p), tele_bound)
        return heuristic_cache[p]

    dist = {initial: 0}
    parent = {}
    heap = [(heuristic(initial[0]), 0, initial)]
    best = None
    best_cost = float("inf")
    expansions = 0

    def insert(state, cost, old, edge):
        # Refunds are represented by the collected-bonus mask.
        net = cost - 10 * (state[7].bit_count() - start_bonus_count)
        if net > remaining:
            return
        if cost < dist.get(state, float("inf")):
            dist[state] = cost
            parent[state] = (old, edge)
            heapq.heappush(heap, (cost + heuristic(state[0]), cost, state))

    while heap:
        f, cost, state = heapq.heappop(heap)
        if cost != dist.get(state):
            continue
        if f >= best_cost:
            break
        expansions += 1
        if expansions % 256 == 0:
            if time.perf_counter() >= deadline or expansions > 160000:
                break
        p, big, direction, alive, li, si, g, bonus = state

        for d, (q, wins) in enumerate(_moves(world, p, big, alive)):
            if wins:
                if cost + 1 < best_cost:
                    best_cost = cost + 1
                    best = (state, ("move", d))
                continue
            nb = bonus | int(world["bonus"][0][q])
            nxt = (q, big, d, alive, li, si, g, nb)
            if nxt != state:
                insert(nxt, cost + 1, state, ("move", d))

        for spell in spells:
            path = paths[g, spell]
            if not path:
                continue
            charge = len(path) + 1
            q, b, al, L, S, nb = p, big, alive, li, si, bonus
            if spell == "size":
                if big:
                    b = 0
                else:
                    q = _grow(world, p, alive)
                    if q is None:
                        continue
                    b = 1
                nb |= int(world["bonus"][b][q])
            elif spell == "tele":
                points = world["large"] if big else world["small"]
                if not points:
                    continue
                index = li if big else si
                index %= len(points)
                x, y = points[index]
                q = y * 64 + x
                if big:
                    L = (index + 1) % len(points)
                else:
                    S = (index + 1) % len(points)
            else:
                if direction == 4:
                    continue
                target = _fire_target(world, p, big, direction, alive)
                if target < 0:
                    continue
                al &= ~world["clear"][target]
            nxt = (q, b, direction, al, L, S, 0, nb)
            if nxt != state:
                insert(nxt, cost + charge, state, ("spell", spell))

    if best is None:
        return None
    state, final_edge = best
    route = [(final_edge, None)]
    while state != initial:
        old, edge = parent[state]
        route.append((edge, state))
        state = old
    route.reverse()
    return route


def _expand_route(world, initial, grid, route):
    queue = []
    state = initial
    current_grid = grid
    for edge, successor in route:
        kind, value = edge
        if kind == "move":
            expected = successor if successor is not None else state
            queue.append(("ACTION" + str(value + 1), None, None,
                          expected, current_grid))
            if successor is not None:
                state = successor
        else:
            path = _spell_path(current_grid, value, world["spells"])
            if not path:
                return []
            for j, bit in enumerate(path):
                current_grid ^= 1 << bit
                expected = state
                if j == len(path) - 1:
                    current_grid = 0
                    expected = successor
                queue.append(("ACTION6", 50 + 5 * (bit // 3),
                              25 + 5 * (bit % 3), expected, current_grid))
            state = successor
    return queue


def _bar(obs):
    return sum(int(obs[2 * i, 62]) == 0 for i in range(32))


def _budget_candidates(k):
    result = []
    for limit in range(30, 121):
        low = (k * limit + 31) // 32
        high = min(limit, ((k + 1) * limit + 31) // 32 - 1)
        for used in range(low, high + 1):
            if used * 32 // limit == k:
                result.append((limit, used))
    return result


def _update_budget(obs, memory, grid, bonus, valid):
    k = _bar(obs)
    candidates = memory.get("budgets")
    last = memory.get("last_action")
    if candidates is None:
        candidates = _budget_candidates(k)
    elif last is not None:
        old_grid = memory.get("before_grid", 0)
        old_bonus = memory.get("before_bonus", 0)
        refund = 10 if (bonus & ~old_bonus) else 0
        name, row, col = last
        transforms = []
        if name == "ACTION6":
            if grid == old_grid and not valid:
                transforms = [(0, 0, 0)]
            else:
                bit = ((row - 50) // 5) * 3 + (col - 25) // 5
                resulting = old_grid ^ (1 << bit)
                cast = resulting in {
                    _PATTERNS[s] for s in memory["world"]["spells"]}
                transforms = [(1, 1 if cast else 0, refund)]
        else:
            transforms = [(0, 1, refund)]
            if not valid:
                transforms.append((0, 0, 0))
        updated = set()
        for limit, used in candidates:
            for pre, post, rebate in transforms:
                new_used = max(0, used + pre - rebate) + post
                if new_used * 32 // limit == k:
                    updated.add((limit, new_used))
        candidates = list(updated) or _budget_candidates(k)
    memory["budgets"] = candidates
    default = 50 if len(memory["world"]["spells"]) <= 1 else 65
    if not candidates:
        return max(1, default * (32 - k) // 32)
    chosen_limit = min({L for L, used in candidates},
                       key=lambda L: (abs(L - default), L))
    used = max(u for L, u in candidates if L == chosen_limit)
    return max(0, chosen_limit - used)


def _reset_or_fallback(available, memory):
    resets = memory.get("resets", 0)
    if "RESET" in available and resets < 2:
        memory.clear()
        memory["resets"] = resets + 1
        return "RESET", None, None
    for action in ("ACTION1", "ACTION3", "ACTION2", "ACTION4"):
        if action in available:
            return action, None, None
    return "ACTION6", 50, 30


def act(obs, available, memory):
    started = time.perf_counter()
    obs = np.asarray(obs)
    world = memory.get("world")
    player, goals = _locate(obs, world["goals"] if world else ())
    if player is None or not goals:
        return _reset_or_fallback(available, memory)

    if world is None:
        world = _build_world(obs, player, goals)
        memory["world"] = world
        memory["queue"] = []

    grid = _grid(obs)
    expected = memory.get("expected")
    px, py, scale, rendered_direction = player
    p, big = py * 64 + px, int(scale == 2)

    if expected is None:
        direction = 4  # Initial facing is not always the internal firing direction.
        alive = world["alive"]
        li, si, bonus = world["initial_large"], 0, 0
    else:
        _, _, direction, alive, li, si, _, bonus = expected

    alive, bonus = _observe_dynamic(obs, world, player, alive, bonus)
    active = [(x + 1, y + 1)
              for x, y in _corners(obs, 11, 5, world["ui"])]
    if big and active:
        point = active[0]
        if point not in world["large"]:
            world["large"].append(point)
            memory["queue"] = []
        li = world["large"].index(point)

    if direction != 4 and rendered_direction != direction:
        direction = rendered_direction

    state = (p, big, direction, alive, li, si, int(grid != 0), bonus)
    valid = expected is not None
    if valid:
        valid = (memory.get("expected_grid") == grid and
                 all(state[i] == expected[i] for i in (0, 1, 2, 3, 4, 5, 7)))

    remaining = _update_budget(obs, memory, grid, bonus, valid)
    if not valid:
        memory["queue"] = []
        if expected is not None:
            # A movement observation establishes the actual facing.
            last = memory.get("last_action", ("", None, None))[0]
            if last in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
                state = (p, big, rendered_direction, alive, li, si,
                         int(grid != 0), bonus)

    queue = memory.get("queue", [])
    if not queue:
        route = _search(world, state, grid, remaining,
                        started + 7.4)
        if route is None and time.perf_counter() < started + 5.0:
            # The exact level budget may not yet be identifiable from its bar.
            route = _search(world, state, grid, max(remaining, 100),
                            started + 7.8)
        if route is None:
            return _reset_or_fallback(available, memory)
        queue = _expand_route(world, state, grid, route)
        if not queue:
            return _reset_or_fallback(available, memory)

    action, row, col, predicted, predicted_grid = queue.pop(0)
    if action not in available:
        return _reset_or_fallback(available, memory)

    memory["queue"] = queue
    memory["expected"] = predicted
    memory["expected_grid"] = predicted_grid
    memory["before_grid"] = grid
    memory["before_bonus"] = bonus
    memory["last_action"] = (action, row, col)
    return action, row, col

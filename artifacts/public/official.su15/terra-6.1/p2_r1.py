import math
import heapq
import numpy as np

_COLORS = (10, 6, 15, 11, 12, 8, 9, 7, 14)
_SIZES = (1, 2, 3, 4, 5, 7, 8, 9, 10)
_PATTERNS = {
    1: ((2, 0), (1, 1), (3, 1), (0, 2),
        (4, 2), (1, 3), (2, 3), (3, 3)),
    2: ((2, 0), (1, 1), (3, 1), (0, 2),
        (2, 2), (4, 2), (1, 3), (3, 3)),
    3: ((0, 0), (2, 0), (4, 0), (1, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
        (1, 3), (2, 3), (3, 3)),
}
_MON_COLOR = {1: 7, 2: 14, 3: 13}


def _rect_overlap(a, b):
    return (a[0] < b[0] + b[2] and b[0] < a[0] + a[2]
            and a[1] < b[1] + b[3] and b[1] < a[1] + a[3])


def _covered(x, y, rects):
    return any(rx <= x < rx + w and ry <= y < ry + h
               for rx, ry, w, h in rects)


def _read_objects(obs, y0, y1, x0=0):
    fruits = []
    fruit_rects = []
    for level, (color, size) in enumerate(zip(_COLORS, _SIZES)):
        mask = (obs == color).astype(np.int16)
        integ = np.pad(mask, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = (integ[size:, size:] - integ[:-size, size:]
                - integ[size:, :-size] + integ[:-size, :-size])
        ys, xs = np.where(sums == size * size)
        for y, x in zip(ys.tolist(), xs.tolist()):
            if x < x0 or y < y0 or y + size > y1:
                continue
            rect = (x, y, size, size)
            if any(_rect_overlap(rect, r) for r in fruit_rects):
                continue
            fruits.append((level, x, y))
            fruit_rects.append(rect)

    monsters = []
    mon_rects = []
    for level in (3, 2, 1):
        color = _MON_COLOR[level]
        pattern = _PATTERNS[level]
        pattern_set = set(pattern)
        candidates = []
        for y in range(y0, min(y1, 64) - 3):
            for x in range(x0, 60):
                visible_matches = 0
                hidden = 0
                valid = True
                for dx, dy in pattern:
                    xx, yy = x + dx, y + dy
                    if _covered(xx, yy, fruit_rects):
                        hidden += 1
                    elif obs[yy, xx] == color:
                        visible_matches += 1
                    else:
                        valid = False
                        break
                if not valid or visible_matches < 4:
                    continue

                extra = 0
                for dy in range(4):
                    for dx in range(5):
                        if (dx, dy) in pattern_set:
                            continue
                        xx, yy = x + dx, y + dy
                        if not _covered(xx, yy, fruit_rects):
                            if obs[yy, xx] == color:
                                extra += 1
                if extra <= 1:
                    candidates.append((-visible_matches, extra, hidden, x, y))

        for _, _, _, x, y in sorted(candidates):
            rect = (x, y, 5, 4)
            if any(_rect_overlap(rect, r) for r in mon_rects):
                continue
            monsters.append((level, x, y))
            mon_rects.append(rect)

    return tuple(sorted(fruits)), tuple(sorted(monsters))


def _recover_hidden_monsters(obs, fruits, monsters, predicted):
    if not predicted or predicted[0] != fruits:
        return monsters

    fruit_rects = [(x, y, _SIZES[level], _SIZES[level])
                   for level, x, y in fruits]
    result = list(monsters)
    for level, x, y in predicted[1]:
        if any(_rect_overlap((x, y, 5, 4), (mx, my, 5, 4))
               for _, mx, my in result):
            continue
        hidden = 0
        valid = True
        for dx, dy in _PATTERNS[level]:
            xx, yy = x + dx, y + dy
            if not (0 <= xx < 64 and 0 <= yy < 63):
                valid = False
                break
            if _covered(xx, yy, fruit_rects):
                hidden += 1
            elif obs[yy, xx] != _MON_COLOR[level]:
                valid = False
                break
        if valid and hidden:
            result.append((level, x, y))
    return tuple(sorted(result))


def _read_pits(obs, fruits):
    nine = (obs == 9)
    score = np.zeros((46, 56), dtype=np.int16)
    for dy in range(9):
        for dx in range(9):
            inside = ((dy in (0, 8) and 2 <= dx <= 6)
                      or (dy in (1, 7) and 1 <= dx <= 7)
                      or 2 <= dy <= 6)
            block = nine[10 + dy:56 + dy, dx:56 + dx]
            score += block.astype(np.int16) * (1 if inside else -2)

    possible = []
    for yy, xx in zip(*np.where(score >= 30)):
        x, y = int(xx), int(yy) + 10
        bad = False
        for lev, fx, fy in fruits:
            if lev != 6:
                continue
            ix = max(0, min(x + 9, fx + 8) - max(x, fx))
            iy = max(0, min(y + 9, fy + 8) - max(y, fy))
            if ix * iy > 35:
                bad = True
                break
        if not bad:
            possible.append((int(score[yy, xx]), x, y))

    possible.sort(reverse=True)
    pits = []
    for _, x, y in possible:
        if all(abs(x - px) >= 8 or abs(y - py) >= 8 for px, py in pits):
            pits.append((x, y))
    return tuple(sorted(pits))


def _read_goals(obs):
    fruits, monsters = _read_objects(obs, 0, 10, 24)
    counts = {}
    for level, _, _ in fruits:
        counts[("f", level)] = counts.get(("f", level), 0) + 1
    for level, _, _ in monsters:
        counts[("m", level)] = counts.get(("m", level), 0) + 1
    return tuple(sorted((kind, level, count)
                        for (kind, level), count in counts.items()))


def _center(obj, monster=False):
    level, x, y = obj[:3]
    if monster:
        return x + 2, y + 2
    size = _SIZES[level]
    return x + size // 2, y + size // 2


def _in_pit(x, y, pits):
    return any(px <= x < px + 9 and py <= y < py + 9
               for px, py in pits)


def _won(fruits, monsters, pits, goals):
    for kind, level, count in goals:
        objects = fruits if kind == "f" else monsters
        n = 0
        for obj in objects:
            if obj[0] == level:
                x, y = _center(obj, kind == "m")
                if _in_pit(x, y, pits):
                    n += 1
        if n != count:
            return False
    return bool(goals)


def _touches_click(x, y, ox, oy, w, h):
    dx = max(ox - x, 0, x - (ox + w - 1))
    dy = max(oy - y, 0, y - (oy + h - 1))
    return dx * dx + dy * dy <= 64


def _clamp_position(x, y, w, h):
    return max(0, min(64 - w, x)), max(10, min(64 - h, y))


def _move_toward(pos, target):
    return pos + max(-4, min(4, target - pos))


def _overlap_objects(a, b, ma=False, mb=False):
    wa = 5 if ma else _SIZES[a[0]]
    ha = 4 if ma else wa
    wb = 5 if mb else _SIZES[b[0]]
    hb = 4 if mb else wb
    return _rect_overlap((a[1], a[2], wa, ha), (b[1], b[2], wb, hb))


def _components(objects, monster=False):
    parent = list(range(len(objects)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _overlap_objects(objects[i], objects[j], monster, monster):
                if objects[i][0] != objects[j][0]:
                    return None
                a, b = root(i), root(j)
                parent[b] = a

    groups = {}
    for i in range(len(objects)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def _advance(state, click, pits, goals):
    fruits0, monsters0 = state
    x, y = click

    # Fruit: level, x, y, selected, bite age, bite data.
    fruits = []
    for lev, fx, fy in fruits0:
        s = _SIZES[lev]
        selected = _touches_click(x, y, fx, fy, s, s)
        fruits.append([lev, fx, fy, selected, -1, None])

    # Monster: level, x, y, selected, direction, cooldown.
    monsters = []
    for lev, mx, my in monsters0:
        selected = _touches_click(x, y, mx, my, 5, 4)
        dx, dy = x - (mx + 2), y - (my + 2)
        norm = math.hypot(dx, dy)
        direction = (dx / norm, dy / norm) if norm else (0.0, 0.0)
        monsters.append([lev, mx, my, selected, direction, 0])

    merged_fruit = False
    for tick in range(16):
        paused = any(f[4] >= 4 and f[0] > 0 for f in fruits)

        for f in fruits:
            if not f[3] or f[4] >= 0:
                continue
            s = _SIZES[f[0]]
            cx, cy = f[1] + s // 2, f[2] + s // 2
            nx = _move_toward(cx, x) - s // 2
            ny = _move_toward(cy, y) - s // 2
            f[1], f[2] = _clamp_position(nx, ny, s, s)

        if not paused:
            for m in monsters:
                if not m[3]:
                    continue
                if m[0] == 1:
                    nx = int(round(m[1] + 3.4 * m[4][0]))
                    ny = int(round(m[2] + 3.4 * m[4][1]))
                else:
                    nx = _move_toward(m[1] + 2, x) - 2
                    ny = _move_toward(m[2] + 2, y) - 2
                m[1], m[2] = _clamp_position(nx, ny, 5, 4)

        for f in list(fruits):
            age = f[4]
            if age < 0:
                continue
            ox, oy, dx, dy, px, py, dying = f[5]
            if age < 4:
                offset = -1 if age % 4 == 1 else (1 if age % 4 == 3 else 0)
                nx, ny = ox, oy + offset
            else:
                if age == 4:
                    f[0] -= 1
                px += 2.5 * dx
                py += 2.5 * dy
                nx, ny = int(round(px)), int(round(py))
                f[5] = (ox, oy, dx, dy, px, py, dying)
            s = _SIZES[max(0, f[0])]
            f[1], f[2] = _clamp_position(nx, ny, s, s)
            f[4] += 1
            if dying and f[4] >= 4:
                fruits.remove(f)
            elif not dying and f[4] >= 8:
                f[4] = -1
                f[5] = None

        for m in monsters:
            if m[5] > 0:
                m[5] -= 1

        if tick < 4 and not paused and fruits:
            for m in monsters:
                if m[3] or m[5] > 0:
                    continue
                candidates = [f for f in fruits if f[4] < 0]
                if not candidates:
                    break
                mcx, mcy = m[1] + 2, m[2] + 2
                target = min(
                    candidates,
                    key=lambda f: ((_center(f)[0] - mcx) ** 2
                                   + (_center(f)[1] - mcy) ** 2))
                tcx, tcy = _center(target)
                speed = 2 if m[0] == 3 else 1
                nx = m[1] + speed * ((tcx > mcx) - (tcx < mcx))
                ny = m[2] + speed * ((tcy > mcy) - (tcy < mcy))
                m[1], m[2] = _clamp_position(nx, ny, 5, 4)

            for m in monsters:
                if m[3] or m[5] > 0:
                    continue
                for f in fruits:
                    if f[4] >= 0 or not _overlap_objects(m, f, True, False):
                        continue
                    fc = _center(f)
                    mc = _center(m, True)
                    dx, dy = fc[0] - mc[0], fc[1] - mc[1]
                    norm = math.hypot(dx, dy)
                    if norm:
                        dx, dy = dx / norm, dy / norm
                    else:
                        dx, dy = 0.0, -1.0
                    f[4] = 0
                    f[5] = (f[1], f[2], dx, dy,
                            float(f[1]), float(f[2]), f[0] == 0)
                    m[5] = 9
                    break

        if monsters and (fruits or tick == 3):
            groups = _components(monsters, True)
            if groups is None:
                return None, False
            if any(len(g) > 1 for g in groups):
                new_monsters = []
                for group in groups:
                    if len(group) == 1:
                        new_monsters.append(monsters[group[0]])
                        continue
                    lev = monsters[group[0]][0]
                    if lev >= 3:
                        continue
                    cx = sum(monsters[i][1] + 2 for i in group) // len(group)
                    cy = sum(monsters[i][2] + 2 for i in group) // len(group)
                    nx, ny = _clamp_position(cx - 2, cy - 2, 5, 4)
                    new_monsters.append([lev + 1, nx, ny, False, (0, 0), 4])
                monsters = new_monsters

        if tick >= 3 and not merged_fruit:
            selected = [f for f in fruits if f[3]]
            groups = _components(selected)
            if groups is None:
                return None, False
            for group in groups:
                if len(group) < 2:
                    continue
                members = [selected[i] for i in group]
                lev = members[0][0]
                cx = sum(_center(f)[0] for f in members) // len(members)
                cy = sum(_center(f)[1] for f in members) // len(members)
                for f in members:
                    if f in fruits:
                        fruits.remove(f)
                if lev < 8:
                    s = _SIZES[lev + 1]
                    nx, ny = _clamp_position(cx - s // 2, cy - s // 2, s, s)
                    fruits.append([lev + 1, nx, ny, False, -1, None])
            merged_fruit = True

        ft = tuple(sorted((f[0], f[1], f[2]) for f in fruits))
        mt = tuple(sorted((m[0], m[1], m[2]) for m in monsters))
        if _won(ft, mt, pits, goals):
            return (ft, mt), True
        if tick >= 3 and all(f[4] < 0 for f in fruits):
            return (ft, mt), False

    return None, False


def _pit_distance(obj, pits, monster=False):
    x, y = _center(obj, monster)
    return min(max(max(px - x, 0, x - (px + 8)),
                   max(py - y, 0, y - (py + 8)))
               for px, py in pits)


def _heuristic(state, pits, goals):
    fruits, monsters = state
    score = 0.0
    for kind, target, count in goals:
        objects = fruits if kind == "f" else monsters
        choices = []
        for obj in objects:
            lev = obj[0]
            distance = _pit_distance(obj, pits, kind == "m") / 7.0
            if kind == "f":
                if lev <= target:
                    weight = 2.0 ** (lev - target)
                    cost = distance + 1.25 * (target - lev)
                else:
                    weight = 1.0
                    cost = distance + 2.5 * (lev - target)
            else:
                weight = 1.0 if lev >= target else 0.5 ** (target - lev)
                cost = distance + 2.0 * abs(target - lev)
            choices.append((cost, weight))
        choices.sort()
        remaining = float(count)
        for cost, weight in choices:
            take = min(remaining, weight)
            score += take * cost
            remaining -= take
            if remaining <= 1e-9:
                break
        score += 30.0 * remaining
    return score


def _candidate_clicks(state, pits):
    fruits, monsters = state
    entities = [(f, False) for f in fruits] + [(m, True) for m in monsters]
    clicks = set()

    def add(x, y):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < 64 and 10 <= y < 63:
            clicks.add((x, y))

    for obj, is_monster in entities:
        cx, cy = _center(obj, is_monster)
        w = 5 if is_monster else _SIZES[obj[0]]
        h = 4 if is_monster else w
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for px, py in pits:
            directions.append((px + 4 - cx, py + 4 - cy))
            add(px + 4, py + 4)
        for other, other_monster in entities:
            if other is obj:
                continue
            ox, oy = _center(other, other_monster)
            if max(abs(ox - cx), abs(oy - cy)) <= 27:
                directions.append((ox - cx, oy - cy))
                add((cx + ox) / 2, (cy + oy) / 2)
                add((2 * cx + ox) / 3, (2 * cy + oy) / 3)
        for dx, dy in directions:
            norm = math.hypot(dx, dy)
            if norm == 0:
                continue
            ux, uy = dx / norm, dy / norm
            best = (cx, cy)
            for d in range(1, 24):
                tx, ty = int(round(cx + ux * d)), int(round(cy + uy * d))
                if not (0 <= tx < 64 and 10 <= ty < 63):
                    break
                if not _touches_click(tx, ty, obj[1], obj[2], w, h):
                    break
                best = (tx, ty)
                if d >= norm:
                    break
            add(*best)
            add(cx + ux * 4, cy + uy * 4)

    if monsters:
        add(0, 10)
        add(63, 10)
        add(0, 62)
        add(63, 62)
    return sorted(clicks)


def _plan(initial, pits, goals, seen, banned):
    initial_h = _heuristic(initial, pits, goals)
    beam = [(initial_h, initial, ())]
    best = None
    best_value = float("inf")
    visited = {initial: 0}
    simulations = 0
    required_mass = sum((1 << level) * count
                        for kind, level, count in goals if kind == "f")

    for depth in range(1, 11):
        next_beam = []
        for _, state, path in beam:
            for click in _candidate_clicks(state, pits):
                if depth == 1 and click in banned:
                    continue
                simulations += 1
                nxt, won = _advance(state, click, pits, goals)
                if nxt is None:
                    continue
                new_path = path + (click,)
                if won:
                    return new_path
                if nxt == state:
                    continue
                if sum(1 << f[0] for f in nxt[0]) < required_mass:
                    continue
                if visited.get(nxt, 1000) <= depth:
                    continue
                visited[nxt] = depth
                h = _heuristic(nxt, pits, goals)
                repeat_penalty = 0.6 * seen.get(nxt, 0) if depth == 1 else 0.0
                value = h + 0.16 * depth + repeat_penalty
                if value < best_value:
                    best_value = value
                    best = new_path
                next_beam.append((value, nxt, new_path))
            if simulations >= 8500:
                break
        if not next_beam or simulations >= 8500:
            break
        beam = heapq.nsmallest(20, next_beam, key=lambda item: item[0])

    return best


def _clear_pending(memory):
    memory["predicted"] = None
    memory["plan"] = ()
    memory["last_state"] = None
    memory["last_board"] = None
    memory["last_click"] = None
    memory["last_bar"] = None


def act(obs, available, memory):
    obs = np.asarray(obs)
    fruits, monsters = _read_objects(obs, 10, 63)
    monsters = _recover_hidden_monsters(
        obs, fruits, monsters, memory.get("predicted"))
    state = (fruits, monsters)

    goals_now = _read_goals(obs)
    if goals_now:
        memory["goals"] = goals_now
    goals = memory.get("goals")

    pits_now = _read_pits(obs, fruits)
    if "pits" not in memory:
        memory["pits"] = pits_now
    elif pits_now:
        pits_list = list(memory["pits"])
        for p in pits_now:
            if all(abs(p[0] - q[0]) >= 8 or abs(p[1] - q[1]) >= 8
                   for q in pits_list):
                pits_list.append(p)
        memory["pits"] = tuple(sorted(pits_list))
    pits = memory.get("pits", ())

    board = obs[:63].tobytes()
    bar = 0
    for value in obs[63]:
        if value != 0:
            break
        bar += 1

    board_bans = memory.setdefault("board_bans", {})
    last_board = memory.get("last_board")
    last_click = memory.get("last_click")
    last_bar = memory.get("last_bar")

    if last_board is not None and last_click is not None:
        unchanged = last_board == board
        penalized = (last_bar is not None and last_bar - bar >= 4)
        if unchanged or penalized:
            board_bans.setdefault(last_board, set()).add(last_click)
            memory["plan"] = ()

    if not goals or not pits or not fruits:
        if "RESET" in available:
            _clear_pending(memory)
            memory["resets"] = memory.get("resets", 0) + 1
            return ("RESET", None, None)
        return ("ACTION6", 10, 0)

    # Preserve enough budget to retry from a clean level if penalties accumulate.
    if bar <= 16 and memory.get("resets", 0) < 2 and "RESET" in available:
        _clear_pending(memory)
        memory["undo_count"] = 0
        memory["resets"] = memory.get("resets", 0) + 1
        return ("RESET", None, None)

    required_mass = sum((1 << level) * count
                        for kind, level, count in goals if kind == "f")
    current_mass = sum(1 << f[0] for f in fruits)
    if current_mass < required_mass:
        if last_board is not None and last_click is not None:
            board_bans.setdefault(last_board, set()).add(last_click)
        undo_count = memory.get("undo_count", 0)
        if "ACTION7" in available and undo_count < 4:
            memory["undo_count"] = undo_count + 1
            _clear_pending(memory)
            return ("ACTION7", None, None)
        if "RESET" in available:
            memory["undo_count"] = 0
            memory["resets"] = memory.get("resets", 0) + 1
            _clear_pending(memory)
            return ("RESET", None, None)
    else:
        memory["undo_count"] = 0

    seen = memory.setdefault("seen", {})
    seen[state] = seen.get(state, 0) + 1
    banned = board_bans.get(board, set())

    plan = None
    if memory.get("predicted") == state:
        plan = memory.get("plan")
    if plan and plan[0] in banned:
        plan = None
    if not plan:
        plan = _plan(state, pits, goals, seen, banned)

    if not plan:
        if "ACTION7" in available and memory.get("last_state") is not None:
            if last_board is not None and last_click is not None:
                board_bans.setdefault(last_board, set()).add(last_click)
            _clear_pending(memory)
            return ("ACTION7", None, None)
        if "RESET" in available:
            memory["resets"] = memory.get("resets", 0) + 1
            _clear_pending(memory)
            return ("RESET", None, None)
        return ("ACTION6", 10, 0)

    click = plan[0]
    predicted, won = _advance(state, click, pits, goals)
    memory["predicted"] = predicted
    memory["plan"] = plan[1:] if not won else ()
    memory["last_state"] = state
    memory["last_board"] = board
    memory["last_click"] = click
    memory["last_bar"] = bar
    return ("ACTION6", int(click[1]), int(click[0]))

from collections import Counter, deque
import heapq
import itertools
import time


DIRS = [
    ("ACTION1", 0, -3),
    ("ACTION2", 0, 3),
    ("ACTION3", -3, 0),
    ("ACTION4", 3, 0),
]


def observe(obs, memory):
    goals = memory.setdefault("goals", {})
    stations = memory.setdefault("stations", {})
    gear = memory.setdefault("gear", set())

    for y in range(1, 62):
        for x in range(1, 63):
            c = int(obs[y, x])
            if c in (0, 1, 2, 4, 5, 15):
                continue
            if all(
                int(obs[y + dy, x + dx]) == 4
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if dx or dy
            ):
                goals[(x, y)] = c

    for size in (5, 6):
        for y in range(64 - size):
            for x in range(65 - size):
                if (int(obs[y, x]) != 2 or
                    int(obs[y, x + size - 1]) != 2 or
                    int(obs[y + size - 1, x]) != 2 or
                    int(obs[y + size - 1, x + size - 1]) != 2):
                    continue

                border = []
                for i in range(size):
                    border.extend((
                        int(obs[y, x + i]),
                        int(obs[y + size - 1, x + i]),
                        int(obs[y + i, x]),
                        int(obs[y + i, x + size - 1]),
                    ))
                if sum(v == 2 for v in border) < len(border) - 2:
                    continue

                inside = Counter(
                    int(obs[yy, xx])
                    for yy in range(y + 1, y + size - 1)
                    for xx in range(x + 1, x + size - 1)
                    if int(obs[yy, xx]) not in (0, 2, 4, 5)
                )
                if inside:
                    color, count = inside.most_common(1)[0]
                    if count >= max(3, (size - 2) ** 2 // 2):
                        stations[(x, y, size, size)] = color

    for y in range(63):
        for x in range(64):
            if int(obs[y, x]) == 1:
                gear.add((x, y))


def zeros(obs):
    return [
        (x, y)
        for y in range(63)
        for x in range(64)
        if int(obs[y, x]) == 0
    ]


def transition_samples(old, new, center, dx, dy):
    cx, cy = center
    samples = {}
    counts = Counter()

    for y in range(63):
        ny = y + dy
        if not 0 <= ny < 63:
            continue
        for x in range(64):
            nx = x + dx
            if not 0 <= nx < 64:
                continue
            c = int(old[y, x])
            if c in (0, 1, 2, 4, 5, 15):
                continue
            if int(new[ny, nx]) != c:
                continue
            if int(new[y, x]) == c and int(old[ny, nx]) == c:
                continue
            samples.setdefault(c, set()).add((x - cx, y - cy))
            counts[c] += 1

    return samples, counts


def complete_mask(points):
    points = set(points)
    if not points:
        return set()

    groups = {}
    for x, y in points:
        groups.setdefault(("h", y), []).append(x)
        groups.setdefault(("v", x), []).append(y)
        groups.setdefault(("d", x - y), []).append(x)
        groups.setdefault(("a", x + y), []).append(x)

    filled = set(points)
    for (kind, key), values in groups.items():
        if len(values) < 3:
            continue
        lo, hi = min(values), max(values)
        if hi - lo > 63:
            continue
        for t in range(lo, hi + 1):
            if kind == "h":
                filled.add((t, key))
            elif kind == "v":
                filled.add((key, t))
            elif kind == "d":
                filled.add((t, t - key))
            else:
                filled.add((t, key - t))

    components = []
    remaining = set(filled)
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    p = (x + dx, y + dy)
                    if p in remaining:
                        remaining.remove(p)
                        component.add(p)
                        stack.append(p)
        components.append(component)

    components.sort(key=len, reverse=True)
    main = components[0]
    for component in components[1:]:
        if len(component) >= 4 and any(
            max(abs(x - xx), abs(y - yy)) <= 3
            for x, y in component
            for xx, yy in main
        ):
            main |= component
    return main


def begin_sprite(memory, center):
    memory["work"] = {
        "center": center,
        "samples": {},
        "last_counts": {},
        "moves": 0,
        "tried": set(),
    }
    memory["phase"] = "explore"


def explore_action(memory, obs):
    work = memory["work"]
    cx, cy = work["center"]

    if work["moves"] == 0:
        preferred = [0, 1, 3, 2] if cy >= 15 else [1, 0, 3, 2]
    else:
        preferred = [3, 2, 0, 1] if cx <= 48 else [2, 3, 0, 1]

    for index in preferred:
        if index in work["tried"]:
            continue
        name, dx, dy = DIRS[index]
        if 0 <= cx + dx < 64 and 0 <= cy + dy < 63:
            work["tried"].add(index)
            memory["pending"] = (obs.copy(), (cx, cy), dx, dy)
            return name, None, None
    return None


def finish_sprite(memory):
    work = memory["work"]
    samples = work["samples"]
    last = work["last_counts"]
    all_points = set()
    totals = Counter()

    for color, points in samples.items():
        all_points.update(points)
        totals[color] += len(points)

    if last and max(last.values()) >= 3:
        color = max(last, key=last.get)
    elif totals:
        color = max(totals, key=totals.get)
    else:
        color = 8

    if memory.get("gear") and color in samples:
        points = samples[color]
    else:
        points = all_points

    mask = complete_mask(points)
    if not mask:
        mask = {(0, 0)}

    memory.setdefault("sprites", []).append({
        "center": work["center"],
        "color": color,
        "mask": mask,
        "samples": set(points),
    })
    memory["phase"] = "switch"
    return "ACTION5", None, None


def path_to(previous, start, end):
    actions = []
    while end != start:
        if end not in previous:
            return []
        end, action = previous[end]
        actions.append(action)
    actions.reverse()
    return actions


def ordinary_paths(sprite, goals, stations, gear):
    cx, cy = sprite["center"]
    initial_color = sprite["color"]
    mask = sprite["mask"]
    collision_mask = set(mask)
    collision_mask.add((0, 0))
    goal_positions = {(x, y) for x, y, _ in goals}
    geometry = {}

    def geometry_at(x, y):
        key = (x, y)
        if key in geometry:
            return geometry[key]

        blocked = any(
            (x + ox, y + oy) in gear
            for ox, oy in collision_mask
        )
        pad = None
        for sx, sy, w, h, color in stations:
            if any(
                sx <= x + ox < sx + w and
                sy <= y + oy < sy + h
                for ox, oy in collision_mask
            ):
                pad = color
                break

        hits = [
            (i, gc)
            for i, (gx, gy, gc) in enumerate(goals)
            if (gx - x, gy - y) in mask
        ]
        safe = (x, y) not in goal_positions
        geometry[key] = blocked, pad, hits, safe
        return geometry[key]

    start = (cx, cy, initial_color)
    queue = deque([start])
    distance = {start: 0}
    previous = {}
    options = {}

    while queue:
        state = queue.popleft()
        x, y, color = state
        cost = distance[state]
        _, _, hits, safe = geometry_at(x, y)

        good = 0
        wrong = 0
        for i, target_color in hits:
            if color == target_color:
                good |= 1 << i
            else:
                wrong |= 1 << i

        key = (good, wrong, safe)
        old = options.get(key)
        if old is None or cost < old[0]:
            options[key] = (cost, state)

        for action, dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < 64 and 0 <= ny < 63):
                continue
            blocked, pad, _, _ = geometry_at(nx, ny)
            if blocked:
                continue
            nxt = (nx, ny, color if pad is None else pad)
            if nxt not in distance:
                distance[nxt] = cost + 1
                previous[nxt] = (state, action)
                queue.append(nxt)

    return options, previous, start


def layer_order(sprites, obs):
    n = len(sprites)
    best_order = list(range(n))
    best_score = -1

    for first in range(n):
        order = [(first + i) % n for i in range(n)]
        top = {}
        count = Counter()
        for index in order:
            sprite = sprites[index]
            cx, cy = sprite["center"]
            for ox, oy in sprite["mask"]:
                p = (cx + ox, cy + oy)
                if 0 <= p[0] < 64 and 0 <= p[1] < 63:
                    count[p] += 1
                    top[p] = sprite["color"]

        score = sum(
            1 for (x, y), number in count.items()
            if number >= 2 and int(obs[y, x]) == top[(x, y)]
        )
        if score > best_score:
            best_score = score
            best_order = order

    return best_order


def choose_positions(sprites, goals, results, obs):
    n = len(sprites)
    full = (1 << len(goals)) - 1
    dp = {(0, False): (0, {})}

    for index in range(n):
        choices = [
            (good, safe, cost, state)
            for (good, wrong, safe), (cost, state)
            in results[index][0].items()
            if wrong == 0
        ]
        next_dp = {}
        for (covered, has_safe), (total, assignment) in dp.items():
            for good, safe, cost, state in choices:
                key = (covered | good, has_safe or safe)
                new_cost = total + cost
                if key not in next_dp or new_cost < next_dp[key][0]:
                    new_assignment = dict(assignment)
                    new_assignment[index] = (state, safe)
                    next_dp[key] = (new_cost, new_assignment)
        dp = next_dp
        if not dp:
            break

    if (full, True) in dp:
        return dp[(full, True)][1]

    dp = {(0, False): (0, {})}
    for index in layer_order(sprites, obs):
        choices = [
            (good, wrong, safe, cost, state)
            for (good, wrong, safe), (cost, state)
            in results[index][0].items()
        ]
        if len(choices) > 500:
            choices.sort(
                key=lambda a: (
                    -(a[0].bit_count() - a[1].bit_count()), a[3]
                )
            )
            choices = choices[:500]

        next_dp = {}
        for (covered, has_safe), (total, assignment) in dp.items():
            for good, wrong, safe, cost, state in choices:
                key = ((covered & ~wrong) | good, has_safe or safe)
                new_cost = total + cost
                if key not in next_dp or new_cost < next_dp[key][0]:
                    new_assignment = dict(assignment)
                    new_assignment[index] = (state, safe)
                    next_dp[key] = (new_cost, new_assignment)

        if len(next_dp) > 12000:
            items = sorted(
                next_dp.items(),
                key=lambda item: (-item[0][0].bit_count(), item[1][0]),
            )[:12000]
            next_dp = dict(items)
        dp = next_dp
        if not dp:
            return None

    if (full, True) in dp:
        return dp[(full, True)][1]

    candidates = [
        (covered.bit_count(), -cost, assignment)
        for (covered, safe), (cost, assignment) in dp.items()
        if safe and covered
    ]
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    return None


def ordinary_plan(memory, obs, selected):
    sprites = memory.get("sprites", [])
    goals = [
        (x, y, color)
        for (x, y), color in sorted(memory.get("goals", {}).items())
    ]
    if not sprites or not goals:
        return []

    stations = [
        (x, y, w, h, color)
        for (x, y, w, h), color in memory.get("stations", {}).items()
    ]
    gear = memory.get("gear", set())
    results = [
        ordinary_paths(sprite, goals, stations, gear)
        for sprite in sprites
    ]
    assignment = choose_positions(sprites, goals, results, obs)
    if not assignment or len(assignment) != len(sprites):
        return []

    n = len(sprites)
    plan = []
    current = selected
    for step in range(n):
        index = (selected + step) % n
        if current != index:
            plan.append("ACTION5")
            current = (current + 1) % n
        state, _ = assignment[index]
        _, previous, start = results[index]
        plan.extend(path_to(previous, start, state))

    for _ in range(n):
        if assignment[current][1]:
            break
        plan.append("ACTION5")
        current = (current + 1) % n
    return plan


def infer_shape(sprite):
    points = sprite.get("samples") or sprite["mask"]
    if len(points) < 8:
        return None

    rows = Counter(y for x, y in points)
    cols = Counter(x for x, y in points)
    max_row = max(rows.values())
    max_col = max(cols.values())
    hw = max(abs(x) for x, y in points)
    hh = max(abs(y) for x, y in points)
    if hw < 2 or hh < 2 or hw > 31 or hh > 31:
        return None

    long_rows = [
        y for y, count in rows.items()
        if count >= max(4, int(max_row * 0.55))
    ]
    long_cols = [
        x for x, count in cols.items()
        if count >= max(4, int(max_col * 0.55))
    ]
    row = max(rows, key=rows.get)
    col = max(cols, key=cols.get)

    separated_rows = (
        len(long_rows) >= 2 and
        max(long_rows) - min(long_rows) >= max(3, hh)
    )
    separated_cols = (
        len(long_cols) >= 2 and
        max(long_cols) - min(long_cols) >= max(3, hw)
    )
    boundary_corner = (
        abs(row) >= hh - 1 and abs(col) >= hw - 1 and
        max_row >= 5 and max_col >= 5
    )

    if separated_rows or separated_cols or boundary_corner:
        return {"kind": "rect", "w": 2 * hw + 1, "h": 2 * hh + 1}

    return {
        "kind": "cross",
        "w": 2 * hw + 1,
        "h": 2 * hh + 1,
        "u": col,
        "v": row,
    }


def initial_state(sprite, shape):
    if "state" in sprite:
        return sprite["state"]
    x, y = sprite["center"]
    if shape["kind"] == "cross":
        return (x, y, shape["u"], shape["v"], sprite["color"])
    return (x, y, shape["w"], shape["h"], sprite["color"])


def contains(shape, state, px, py, include_center=True):
    x, y, u, v, _ = state
    dx, dy = px - x, py - y
    if include_center and dx == 0 and dy == 0:
        return True

    if shape["kind"] == "cross":
        hw, hh = shape["w"] // 2, shape["h"] // 2
        return (
            (dx == u and -hh <= dy <= hh) or
            (dy == v and -hw <= dx <= hw)
        )

    w, h = u, v
    left, top = -(w // 2), -(h // 2)
    right, bottom = left + w - 1, top + h - 1
    return (
        (left <= dx <= right and (dy == top or dy == bottom)) or
        (top <= dy <= bottom and (dx == left or dx == right))
    )


def pad_contact(shape, state, station):
    x, y, u, v, _ = state
    sx, sy, w, h, _ = station
    ex, ey = sx + w - 1, sy + h - 1

    if shape["kind"] == "cross":
        hw, hh = shape["w"] // 2, shape["h"] // 2
        return (
            (sx <= x + u <= ex and y - hh <= ey and y + hh >= sy) or
            (sy <= y + v <= ey and x - hw <= ex and x + hw >= sx)
        )

    left, top = x - u // 2, y - v // 2
    right, bottom = left + u - 1, top + v - 1
    return (
        ((sx <= left <= ex or sx <= right <= ex) and
         top <= ey and bottom >= sy) or
        ((sy <= top <= ey or sy <= bottom <= ey) and
         left <= ex and right >= sx)
    )


def mechanism_move(shape, state, dx, dy, gear, box, stations):
    x, y, u, v, color = state
    nx, ny = x + dx, y + dy
    if not (0 <= nx < 64 and 0 <= ny < 64):
        return None

    trial = (nx, ny, u, v, color)
    gx0, gy0, gx1, gy1 = box
    if shape["kind"] == "cross":
        hw, hh = shape["w"] // 2, shape["h"] // 2
    else:
        hw, hh = u // 2 + 1, v // 2 + 1

    near = not (
        nx + hw < gx0 or nx - hw > gx1 or
        ny + hh < gy0 or ny - hh > gy1
    )
    collides = near and any(
        contains(shape, trial, gx, gy)
        for gx, gy in gear
    )

    if collides and shape["kind"] == "rect":
        w, h = u, v
        left, top = nx - w // 2, ny - h // 2
        if dx:
            if w <= 6:
                return None
            nw, nh = w - 3, h + 3
            top += h // 2 - nh // 2
            top = round(top / 3) * 3
            if dx < 0:
                left -= dx
            nx, ny = left + nw // 2, top + nh // 2
            u, v = nw, nh
        else:
            if h <= 6:
                return None
            nw, nh = w + 3, h - 3
            left += w // 2 - nw // 2
            left = round(left / 3) * 3
            if dy < 0:
                top -= dy
            nx, ny = left + nw // 2, top + nh // 2
            u, v = nw, nh

    elif collides:
        w, h = shape["w"], shape["h"]
        col = u + w // 2
        row = v + h // 2
        in_x = gx0 <= nx + u <= gx1
        in_y = gy0 <= ny + v <= gy1

        if in_x and in_y:
            return None

        if dx:
            sign = 1 if dx > 0 else -1
            if in_x:
                allowed = col > 0 if sign > 0 else col < w - 2
                if not allowed:
                    return None
                u -= sign * 3
            elif in_y:
                nx, ny = x, y
                allowed = col < w - 2 if sign > 0 else col > 0
                if allowed:
                    u += sign * 3
        else:
            sign = 1 if dy > 0 else -1
            if in_y:
                allowed = row > 0 if sign > 0 else row < h - 2
                if not allowed:
                    return None
                v -= sign * 3
            elif in_x:
                nx, ny = x, y
                allowed = row < h - 2 if sign > 0 else row > 0
                if allowed:
                    v += sign * 3

    result = (nx, ny, u, v, color)
    if result == state:
        return None

    for station in stations:
        if station[4] != color and pad_contact(shape, result, station):
            color = station[4]
            break
    return (nx, ny, u, v, color)


def mechanism_candidates(sprite, shape, group, color, stations):
    start = initial_state(sprite, shape)
    sx, sy, su, sv, initial_color = start

    if initial_color != color and not any(s[4] == color for s in stations):
        return []

    found = set()
    goal_positions = set(group)

    if shape["kind"] == "cross":
        hw, hh = shape["w"] // 2, shape["h"] // 2
        verticals = {x for x, y in group}
        horizontals = {y for x, y in group}

        for axis_x in verticals:
            for axis_y in horizontals:
                if not all(x == axis_x or y == axis_y for x, y in group):
                    continue
                for x in range(sx % 3, 64, 3):
                    u = axis_x - x
                    if not (-hw <= u <= hw and (u - su) % 3 == 0):
                        continue
                    for y in range(sy % 3, 64, 3):
                        v = axis_y - y
                        if not (-hh <= v <= hh and (v - sv) % 3 == 0):
                            continue
                        if (x, y) in goal_positions:
                            continue
                        state = (x, y, u, v, color)
                        if all(
                            contains(shape, state, gx, gy, False)
                            for gx, gy in group
                        ):
                            found.add(state)

    else:
        total = su + sv
        start_width = su - 3 * ((su - 4) // 3)
        left_mods = {(sx - su // 2) % 3, 0}
        top_mods = {(sy - sv // 2) % 3, 0}

        for w in range(start_width, total - 3, 3):
            h = total - w
            if w < 4 or h < 4:
                continue
            lefts = {
                left
                for gx, gy in group
                for left in (gx, gx - w + 1)
            }
            tops = {
                top
                for gx, gy in group
                for top in (gy, gy - h + 1)
            }
            for left in lefts:
                if left % 3 not in left_mods:
                    continue
                for top in tops:
                    if top % 3 not in top_mods:
                        continue
                    x, y = left + w // 2, top + h // 2
                    if not (0 <= x < 64 and 0 <= y < 64):
                        continue
                    if (x, y) in goal_positions:
                        continue
                    state = (x, y, w, h, color)
                    if all(
                        contains(shape, state, gx, gy, False)
                        for gx, gy in group
                    ):
                        found.add(state)

    def estimate(state):
        x, y, u, v, _ = state
        return (
            abs(x - sx) + abs(y - sy) +
            2 * (abs(u - su) + abs(v - sv))
        )

    return sorted(found, key=estimate)[:32]


def mechanism_astar(sprite, shape, group, color, candidates,
                    all_goals, stations, gear, box):
    start = initial_state(sprite, shape)
    if not candidates:
        return None

    deadline = time.monotonic() + 8.3
    target_stations = [s for s in stations if s[4] == color]
    ticket = itertools.count()
    goal_positions = {(x, y) for x, y, _ in all_goals}

    def heuristic(state):
        x, y, u, v, c = state
        best = 10 ** 9
        for tx, ty, tu, tv, _ in candidates:
            dist = (abs(x - tx) + abs(y - ty)) // 3
            change = (abs(u - tu) + abs(v - tv)) // 3
            if shape["kind"] == "rect":
                change //= 2
            best = min(best, dist + 2 * change)

        if c != color and target_stations:
            if shape["kind"] == "cross":
                reach = max(shape["w"], shape["h"]) // 2
            else:
                reach = max(u, v) // 2
            pad_distance = min(
                max(
                    0,
                    abs(x - (sx + w // 2)) +
                    abs(y - (sy + h // 2)) - reach
                ) // 3
                for sx, sy, w, h, _ in target_stations
            )
            best += pad_distance
        return best

    def is_goal(state):
        x, y, _, _, c = state
        if c != color or (x, y) in goal_positions:
            return False
        if not all(
            contains(shape, state, gx, gy, False)
            for gx, gy in group
        ):
            return False
        return not any(
            gc != color and contains(shape, state, gx, gy, False)
            for gx, gy, gc in all_goals
        )

    queue = [(2 * heuristic(start), 0, next(ticket), start)]
    distance = {start: 0}
    previous = {}
    expanded = 0

    while queue:
        _, cost, _, state = heapq.heappop(queue)
        if distance.get(state) != cost:
            continue
        if is_goal(state):
            return path_to(previous, start, state), state

        expanded += 1
        if expanded % 256 == 0 and time.monotonic() >= deadline:
            break
        if expanded > 60000:
            break

        for action, dx, dy in DIRS:
            nxt = mechanism_move(
                shape, state, dx, dy, gear, box, stations
            )
            if nxt is None:
                continue
            new_cost = cost + 1
            if new_cost >= distance.get(nxt, 10 ** 9):
                continue
            distance[nxt] = new_cost
            previous[nxt] = (state, action)
            heapq.heappush(
                queue,
                (new_cost + 2 * heuristic(nxt),
                 new_cost, next(ticket), nxt),
            )

    return None


def read_selected_shape(obs, shape, predicted, colors):
    x, y, u, v, color = predicted
    visible_zeros = zeros(obs)
    if visible_zeros:
        zx, zy = min(
            visible_zeros,
            key=lambda p: abs(p[0] - x) + abs(p[1] - y)
        )
        if abs(zx - x) + abs(zy - y) <= 12:
            x, y = zx, zy

    colors = set(colors)
    colors.add(color)
    colors -= {0, 1, 2, 4, 5, 15}

    best = None
    if shape["kind"] == "cross":
        hw, hh = shape["w"] // 2, shape["h"] // 2
        for c in colors:
            col_scores = []
            for du in range(-hw, hw + 1):
                xx = x + du
                if not 0 <= xx < 64:
                    continue
                score = sum(
                    1 for yy in range(max(0, y - hh), min(63, y + hh + 1))
                    if int(obs[yy, xx]) == c
                )
                col_scores.append((score, -abs(du - u), du))
            row_scores = []
            for dv in range(-hh, hh + 1):
                yy = y + dv
                if not 0 <= yy < 63:
                    continue
                score = sum(
                    1 for xx in range(max(0, x - hw), min(64, x + hw + 1))
                    if int(obs[yy, xx]) == c
                )
                row_scores.append((score, -abs(dv - v), dv))
            if not col_scores or not row_scores:
                continue
            cs = max(col_scores)
            rs = max(row_scores)
            rank = (
                cs[0] + rs[0],
                int(c == color),
                cs[1] + rs[1],
            )
            if best is None or rank > best[0]:
                best = (rank, (x, y, cs[2], rs[2], c))

    else:
        total = u + v
        for w in range(4, total - 3):
            if (w - u) % 3:
                continue
            h = total - w
            left, top = x - w // 2, y - h // 2
            right, bottom = left + w - 1, top + h - 1
            edge = set()
            for xx in range(left, right + 1):
                edge.add((xx, top))
                edge.add((xx, bottom))
            for yy in range(top, bottom + 1):
                edge.add((left, yy))
                edge.add((right, yy))
            for c in colors:
                score = sum(
                    1 for xx, yy in edge
                    if 0 <= xx < 64 and 0 <= yy < 63
                    and int(obs[yy, xx]) == c
                )
                rank = (score, int(c == color), -abs(w - u))
                if best is None or rank > best[0]:
                    best = (rank, (x, y, w, h, c))

    if best is not None and best[0][0] >= 5:
        return best[1]
    return (x, y, u, v, color)


def prepare_mechanism(memory, selected):
    sprites = memory.get("sprites", [])
    if not sprites:
        return False

    shapes = [infer_shape(sprite) for sprite in sprites]
    if any(shape is None for shape in shapes):
        return False

    goals = [
        (x, y, color)
        for (x, y), color in sorted(memory.get("goals", {}).items())
    ]
    groups = {}
    for x, y, color in goals:
        groups.setdefault(color, []).append((x, y))

    if not groups or len(groups) > len(sprites):
        return False

    stations = [
        (x, y, w, h, color)
        for (x, y, w, h), color in memory.get("stations", {}).items()
    ]
    gear = memory.get("gear", set())
    if not gear:
        return False
    box = (
        min(x for x, y in gear),
        min(y for x, y in gear),
        max(x for x, y in gear),
        max(y for x, y in gear),
    )

    colors = sorted(groups, key=lambda c: -len(groups[c]))
    candidates = {}
    for i, sprite in enumerate(sprites):
        for color in colors:
            candidates[i, color] = mechanism_candidates(
                sprite, shapes[i], groups[color], color, stations
            )

    assignments = []
    for indices in itertools.permutations(range(len(sprites)), len(colors)):
        if not all(candidates[i, color]
                   for i, color in zip(indices, colors)):
            continue
        score = 0
        for i, color in zip(indices, colors):
            state = initial_state(sprites[i], shapes[i])
            target = candidates[i, color][0]
            score += abs(state[0] - target[0]) + abs(state[1] - target[1])
            score += 2 * (
                abs(state[2] - target[2]) + abs(state[3] - target[3])
            )
        assignments.append((score, indices))

    if not assignments:
        return False

    assignments.sort()
    _, indices = assignments[0]
    tasks = {i: color for i, color in zip(indices, colors)}

    memory["mechanism"] = {
        "shapes": shapes,
        "goals": goals,
        "groups": groups,
        "stations": stations,
        "gear": gear,
        "box": box,
        "tasks": tasks,
        "done": set(),
        "retries": {},
        "selected": selected,
        "path": [],
        "final": None,
    }
    memory["phase"] = "mechanism"
    return True


def mechanism_action(memory, obs):
    model = memory["mechanism"]
    sprites = memory["sprites"]
    n = len(sprites)

    if model["path"]:
        return model["path"].pop(0), None, None

    if model["final"] is not None:
        index, predicted = model["final"]
        colors = [s[4] for s in model["stations"]]
        colors += [s["color"] for s in sprites]
        actual = read_selected_shape(
            obs, model["shapes"][index], predicted, colors
        )
        sprites[index]["state"] = actual
        sprites[index]["center"] = (actual[0], actual[1])
        sprites[index]["color"] = actual[4]
        model["final"] = None

        color = model["tasks"][index]
        group = model["groups"][color]
        missing = any(int(obs[y, x]) != color for x, y in group)
        retries = model["retries"].get(index, 0)

        if missing and retries < 3:
            model["retries"][index] = retries + 1
            model["done"].discard(index)
        else:
            model["done"].add(index)

    if len(model["done"]) == len(model["tasks"]):
        for index, color in model["tasks"].items():
            if model["retries"].get(index, 0) >= 3:
                continue
            if any(
                int(obs[y, x]) != color
                for x, y in model["groups"][color]
            ):
                model["done"].discard(index)
                model["retries"][index] = (
                    model["retries"].get(index, 0) + 1
                )
                break

        if len(model["done"]) == len(model["tasks"]):
            memory["phase"] = "fallback"
            return None

    index = model["selected"]
    if index in model["tasks"] and index not in model["done"]:
        color = model["tasks"][index]
        shape = model["shapes"][index]

        if "state" in sprites[index]:
            colors = [s[4] for s in model["stations"]]
            colors += [s["color"] for s in sprites]
            state = read_selected_shape(
                obs, shape, sprites[index]["state"], colors
            )
            sprites[index]["state"] = state
            sprites[index]["center"] = (state[0], state[1])
            sprites[index]["color"] = state[4]

        candidates = mechanism_candidates(
            sprites[index], shape, model["groups"][color],
            color, model["stations"]
        )
        result = mechanism_astar(
            sprites[index], shape, model["groups"][color], color,
            candidates, model["goals"], model["stations"],
            model["gear"], model["box"]
        )
        if result is None:
            memory["phase"] = "fallback"
            return None

        path, state = result
        model["path"] = path
        model["final"] = (index, state)
        if path:
            return model["path"].pop(0), None, None

        model["final"] = None
        sprites[index]["state"] = state
        sprites[index]["center"] = (state[0], state[1])
        sprites[index]["color"] = state[4]
        model["done"].add(index)

    model["selected"] = (index + 1) % n
    return "ACTION5", None, None


def act(obs, available, memory):
    observe(obs, memory)

    if "phase" not in memory:
        selected = zeros(obs)
        if not selected:
            return "ACTION5", None, None
        memory["sprites"] = []
        begin_sprite(memory, selected[0])

    if memory["phase"] == "explore":
        pending = memory.pop("pending", None)
        if pending is not None:
            old, old_center, dx, dy = pending
            expected = (old_center[0] + dx, old_center[1] + dy)
            work = memory["work"]
            samples, counts = transition_samples(
                old, obs, old_center, dx, dy
            )

            visible_center = (
                0 <= expected[0] < 64 and
                0 <= expected[1] < 63 and
                int(obs[expected[1], expected[0]]) == 0
            )
            old_still_selected = (
                0 <= old_center[0] < 64 and
                0 <= old_center[1] < 63 and
                int(obs[old_center[1], old_center[0]]) == 0
            )
            hidden_center = (
                bool(memory.get("gear")) and
                not old_still_selected and counts and
                max(counts.values()) >= 5
            )

            if visible_center or hidden_center:
                for color, points in samples.items():
                    work["samples"].setdefault(color, set()).update(points)
                work["last_counts"] = dict(counts)
                work["center"] = expected
                work["moves"] += 1
                work["tried"] = set()

        if memory["work"]["moves"] >= 2:
            return finish_sprite(memory)

        action = explore_action(memory, obs)
        if action is not None:
            return action
        return finish_sprite(memory)

    if memory["phase"] == "switch":
        selected_pixels = zeros(obs)
        known = [sprite["center"] for sprite in memory.get("sprites", [])]
        new_centers = [p for p in selected_pixels if p not in known]

        if new_centers and len(known) < 8:
            begin_sprite(memory, new_centers[0])
            action = explore_action(memory, obs)
            if action is not None:
                return action

        selected = 0
        for p in selected_pixels:
            if p in known:
                selected = known.index(p)
                break

        if memory.get("gear") and prepare_mechanism(memory, selected):
            action = mechanism_action(memory, obs)
            if action is not None:
                return action
        else:
            memory["plan"] = ordinary_plan(memory, obs, selected)
            memory["phase"] = "execute"

    if memory["phase"] == "mechanism":
        action = mechanism_action(memory, obs)
        if action is not None:
            return action

    if memory["phase"] == "fallback":
        selected_pixels = zeros(obs)
        known = [sprite["center"] for sprite in memory.get("sprites", [])]
        selected = 0
        for p in selected_pixels:
            if p in known:
                selected = known.index(p)
                break
        memory["plan"] = ordinary_plan(memory, obs, selected)
        memory["phase"] = "execute"

    if memory["phase"] == "execute":
        plan = memory.get("plan", [])
        if plan:
            return plan.pop(0), None, None

        attempts = memory.get("relearns", 0)
        if attempts < 2:
            memory["relearns"] = attempts + 1
            selected_pixels = zeros(obs)
            if selected_pixels:
                memory["sprites"] = []
                begin_sprite(memory, selected_pixels[0])
                action = explore_action(memory, obs)
                if action is not None:
                    return action

    return "ACTION5", None, None

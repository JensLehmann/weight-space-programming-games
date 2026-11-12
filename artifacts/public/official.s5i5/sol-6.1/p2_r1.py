import numpy as np
import collections
import heapq
import itertools
import time

DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for yy, xx in zip(*np.nonzero(mask)):
        x, y = int(xx), int(yy)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(x, y)]
        points = []
        while stack:
            x, y = stack.pop()
            points.append((x, y))
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        result.append(points)
    return result


def controls_and_ui(obs):
    mask = obs == 2
    mask[63] = False
    ui = np.zeros((64, 64), dtype=bool)
    controls = []
    for points in components(mask):
        xs, ys = zip(*points)
        x, y = min(xs), min(ys)
        w, h = max(xs) - x + 1, max(ys) - y + 1
        if min(w, h) < 5:
            continue
        patch = obs[y:y+h, x:x+w]
        counts = collections.Counter(
            int(v) for v in patch.ravel()
            if int(v) not in (2, 3, 4, 5, 13, 15))
        if not counts:
            continue
        color = counts.most_common(1)[0][0]
        ui[y:y+h, x:x+w] = True
        if w == h:
            controls.append(("rot", color, y+h//2, x+w//2))
        elif w > h:
            controls.append(("dec", color, y+h//2, x+1))
            controls.append(("inc", color, y+h//2, x+w-2))
        else:
            controls.append(("dec", color, y+1, x+w//2))
            controls.append(("inc", color, y+h-2, x+w//2))
    return controls, ui


def find_marks(obs, ui):
    mask = (obs == 13) & ~ui
    mask[63] = False
    goals = []
    arms = set()
    for y in range(1, 62):
        for x in range(1, 63):
            if (mask[y-1, x] and mask[y+1, x]
                    and mask[y, x-1] and mask[y, x+1]):
                goals.append((x-1, y-1))
                arms.update(((x, y-1), (x, y+1),
                             (x-1, y), (x+1, y)))
    marks = []
    for yy, xx in zip(*np.nonzero(mask)):
        x, y = int(xx), int(yy)
        if (x, y) not in arms:
            marks.append((x-1, y-1))
    return sorted(set(goals)), sorted(set(marks))


def restore_image(obs, ui, goals):
    clean = obs.copy()

    def pair_value(a, b):
        if a == b:
            return int(a)
        if a == 3:
            return int(b)
        if b == 3:
            return int(a)
        return int(a if a != 13 else b)

    for x, y in goals:
        if not (0 <= x <= 61 and 0 <= y <= 60):
            continue
        a, b = clean[y, x], clean[y, x+2]
        c, d = clean[y+2, x], clean[y+2, x+2]
        clean[y, x+1] = pair_value(a, b)
        clean[y+1, x] = pair_value(a, c)
        clean[y+1, x+2] = pair_value(b, d)
        clean[y+2, x+1] = pair_value(c, d)

    for yy, xx in zip(*np.nonzero((clean == 13) & ~ui)):
        x, y = int(xx), int(yy)
        vals = []
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1),
                       (-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < 64 and 0 <= ny < 63:
                v = int(clean[ny, nx])
                if v not in (3, 13):
                    vals.append(v)
        clean[y, x] = (collections.Counter(vals).most_common(1)[0][0]
                       if vals else 5)
    return clean


def parse_bars(clean, ui, colors):
    bars = []
    usable = ~ui
    usable[63] = False
    for color in sorted(colors):
        for points in components((clean == color) & usable):
            xs, ys = zip(*points)
            x, y = min(xs), min(ys)
            w, h = max(xs)-x+1, max(ys)-y+1
            choices = []
            if w == 3 and h % 3 == 2:
                n = (h+1)//3
                if y > 0 and np.all(clean[y-1, x:x+3] == 3):
                    choices.append((color, x, y-1, n, 2))
                if y+h < 63 and np.all(clean[y+h, x:x+3] == 3):
                    choices.append((color, x, y+h-2, n, 0))
            if h == 3 and w % 3 == 2:
                n = (w+1)//3
                if x > 0 and np.all(clean[y:y+3, x-1] == 3):
                    choices.append((color, x-1, y, n, 1))
                if x+w < 64 and np.all(clean[y:y+3, x+w] == 3):
                    choices.append((color, x+w-2, y, n, 3))
            if choices:
                bars.append(choices[0])
                continue

            dims = {(3, 3*max(1, (h+2)//3)),
                    (3*max(1, (w+2)//3), 3)}
            candidates = []
            for ww, hh in dims:
                for xx in range(max(0, x-2), x+1):
                    for yy in range(max(0, y-2), y+1):
                        if xx+ww > 64 or yy+hh > 63:
                            continue
                        if x+w > xx+ww or y+h > yy+hh:
                            continue
                        p = clean[yy:yy+hh, xx:xx+ww]
                        if np.any(ui[yy:yy+hh, xx:xx+ww]):
                            continue
                        if not np.all((p == color) | (p == 3)):
                            continue
                        possible = []
                        if ww == 3:
                            possible.extend((0, 2))
                        if hh == 3:
                            possible.extend((1, 3))
                        for direction in possible:
                            if direction == 0:
                                edge, body = p[-1, :], p[:-1, :]
                                ax, ay = xx, yy+hh-3
                            elif direction == 1:
                                edge, body = p[:, 0], p[:, 1:]
                                ax, ay = xx, yy
                            elif direction == 2:
                                edge, body = p[0, :], p[1:, :]
                                ax, ay = xx, yy
                            else:
                                edge, body = p[:, -1], p[:, :-1]
                                ax, ay = xx+ww-3, yy
                            if np.all(edge == 3) and np.all(body == color):
                                n = max(ww, hh)//3
                                candidates.append((ww*hh, color, ax, ay,
                                                   n, direction))
            if candidates:
                _, c, ax, ay, n, direction = max(candidates)
                bars.append((c, ax, ay, n, direction))
    return sorted(set(bars), key=lambda b: (b[0], b[2], b[1]))


def parse_rigids(clean, ui):
    mask = (clean == 1) & ~ui
    mask[63] = False
    rectangles = []
    active = {}
    for y in range(64):
        runs = []
        x = 0
        while x < 64:
            if not mask[y, x]:
                x += 1
                continue
            start = x
            while x < 64 and mask[y, x]:
                x += 1
            runs.append((start, x-start))
        new_active = {}
        for run in runs:
            if run in active:
                i = active[run]
                xx, yy, w, h = rectangles[i]
                rectangles[i] = (xx, yy, w, h+1)
            else:
                i = len(rectangles)
                rectangles.append((run[0], y, run[1], 1))
            new_active[run] = i
        active = new_active
    return sorted(rectangles, key=lambda r: (r[1], r[0]))


def rectangle(state, i, nb):
    x, y, a, b = state[4*i:4*i+4]
    if i >= nb:
        return x, y, a, b
    n, d = a, b
    if d == 0:
        return x, y-3*(n-1), 3, 3*n
    if d == 1:
        return x, y, 3*n, 3
    if d == 2:
        return x, y, 3, 3*n
    return x-3*(n-1), y, 3*n, 3


def tip(state, i):
    x, y, n, d = state[4*i:4*i+4]
    dx, dy = DIRS[d]
    return x+3*(n-1)*dx, y+3*(n-1)*dy


def overlap(a, b):
    x, y, w, h = a
    xx, yy, ww, hh = b
    return x < xx+ww and xx < x+w and y < yy+hh and yy < y+h


def update_closures(model):
    children = [[] for _ in range(model["n"])]
    for i, parent in enumerate(model["parents"]):
        if parent is not None and parent != i:
            children[parent].append(i)
    descendants = []
    for i in range(model["n"]):
        found = set()
        stack = list(children[i])
        while stack:
            j = stack.pop()
            if j == i or j in found:
                continue
            found.add(j)
            stack.extend(children[j])
        descendants.append(tuple(sorted(found)))
    model["desc"] = descendants
    model["children"] = children
    for action in model["actions"]:
        affected = set(action["roots"])
        for r in action["roots"]:
            affected.update(descendants[r])
        action["affected"] = tuple(sorted(affected))


def update_capabilities(model):
    model["rotatable"] = {
        i for a in model["actions"] if a["kind"] == "rot"
        for i in a["roots"]}
    model["resizable"] = {
        i for a in model["actions"] if a["kind"] != "rot"
        for i in a["roots"]}
    marked = {i for i, _ in model["markers"]}
    # Free-standing, unmarked rotary bars may extend out of the viewport.
    model["exterior"] = {
        i for i in model["rotatable"]
        if model["parents"][i] is None and not model["desc"][i]
        and i not in marked}


def bind_markers(model, state, positions):
    result = []
    for mx, my in positions:
        candidates = []
        for i in range(model["nb"]):
            r = rectangle(state, i, model["nb"])
            if overlap((mx, my, 3, 3), r):
                tx, ty = tip(state, i)
                candidates.append((abs(tx-mx)+abs(ty-my), i))
        if not candidates:
            continue
        _, i = min(candidates)
        tx, ty = tip(state, i)
        dx, dy = mx-tx, my-ty
        direction = state[4*i+3]
        offsets = [None]*4
        for _ in range(4):
            offsets[direction] = (dx, dy)
            dx, dy = dy, -dx
            direction = (direction-1) % 4
        result.append((i, tuple(offsets)))
    model["markers"] = result


def wall_count(model, r, i=None):
    x, y, w, h = r
    if min(w, h) <= 0 or max(w, h) > 75:
        return 1
    if i in model.get("exterior", ()):
        if x < -45 or y < -45 or x+w > 109 or y+h > 109:
            return 1
    elif min(x, y) < 0 or x+w > 64 or y+h > 64:
        return 1

    xx, yy = max(0, x), max(0, y)
    ex, ey = min(64, x+w), min(64, y+h)
    if xx >= ex or yy >= ey:
        return 0
    p = model["integral"]
    return (p[ey*65+ex]-p[yy*65+ex]
            -p[ey*65+xx]+p[yy*65+xx])


def distance_tables(model, state):
    blocked = model["static"].copy()
    for i in range(model["nb"], model["n"]):
        if model["parents"][i] is None:
            x, y, w, h = rectangle(state, i, model["nb"])
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(64, x+w), min(64, y+h)
            if x0 < x1 and y0 < y1:
                blocked[y0:y1, x0:x1] = True
    prefix = np.zeros((65, 65), dtype=np.int32)
    prefix[1:, 1:] = blocked.astype(np.int32).cumsum(0).cumsum(1)

    def free(x, y):
        if x < 0 or y < 0 or x+3 > 64 or y+3 > 64:
            return False
        return (prefix[y+3, x+3]-prefix[y, x+3]
                -prefix[y+3, x]+prefix[y, x]) == 0

    tables = []
    for gx, gy in model["goals"]:
        dist = {}
        if free(gx, gy):
            dist[(gx, gy)] = 0
            queue = collections.deque([(gx, gy)])
            while queue:
                x, y = queue.popleft()
                nd = dist[(x, y)]+1
                for dx, dy in DIRS:
                    xx, yy = x+3*dx, y+3*dy
                    if (xx, yy) not in dist and free(xx, yy):
                        dist[(xx, yy)] = nd
                        queue.append((xx, yy))
        table = []
        for y in range(64):
            for x in range(64):
                manhattan = (abs(x-gx)+abs(y-gy))/3.0
                geo = dist.get((x, y), manhattan+8.0)
                table.append(0.85*geo+0.15*manhattan)
        tables.append(table)
    model["distances"] = tables


def make_model(obs):
    controls, ui = controls_and_ui(obs)
    goals, marks = find_marks(obs, ui)
    clean = restore_image(obs, ui, goals)
    colors = {c for _, c, _, _ in controls if c != 1}
    bars = parse_bars(clean, ui, colors)
    if not bars:
        return None, None
    rigids = parse_rigids(clean, ui)
    nb = len(bars)
    state = []
    for c, x, y, length, direction in bars:
        state.extend((x, y, length, direction))
    for r in rigids:
        state.extend(r)
    state = tuple(state)
    n = nb+len(rigids)
    parents = [None]*n

    for child in range(nb):
        cx, cy = state[4*child:4*child+2]
        possible = []
        for parent in range(nb):
            if child == parent:
                continue
            tx, ty = tip(state, parent)
            dx, dy = DIRS[state[4*parent+3]]
            if (cx, cy) == (tx+3*dx, ty+3*dy):
                possible.append(parent)
        if possible:
            parents[child] = possible[0]

    for child in range(nb, n):
        r = rectangle(state, child, nb)
        for parent in range(nb):
            tx, ty = tip(state, parent)
            dx, dy = DIRS[state[4*parent+3]]
            if overlap((tx+3*dx, ty+3*dy, 3, 3), r):
                parents[child] = parent
                break

    for i in range(n):
        seen = {i}
        p = parents[i]
        while p is not None:
            if p in seen:
                parents[i] = None
                break
            seen.add(p)
            p = parents[p]

    actions = []
    used = set()
    for kind, color, row, col in controls:
        if (kind, color) in used:
            continue
        roots = tuple(i for i, b in enumerate(bars) if b[0] == color)
        if roots:
            actions.append(dict(kind=kind, color=color, row=row, col=col,
                                roots=roots))
            used.add((kind, color))

    static = (clean == 15) & ~ui
    static[63] = False
    prefix = np.zeros((65, 65), dtype=np.int32)
    prefix[1:, 1:] = static.astype(np.int32).cumsum(0).cumsum(1)
    model = dict(nb=nb, n=n, colors=tuple(b[0] for b in bars),
                 parents=parents, actions=actions, goals=goals,
                 ui=ui, ui_pixels=obs.copy(), static=static,
                 integral=prefix.ravel().tolist())
    update_closures(model)
    bind_markers(model, state, marks)
    update_capabilities(model)
    distance_tables(model, state)
    return model, state


def marker_positions(model, state):
    result = []
    for i, offsets in model["markers"]:
        tx, ty = tip(state, i)
        dx, dy = offsets[state[4*i+3]]
        result.append((tx+dx, ty+dy))
    return result


def won(model, state):
    points = set(marker_positions(model, state))
    return bool(model["goals"]) and all(g in points for g in model["goals"])


def marker_heuristic(model, state):
    goals = model["goals"]
    positions = marker_positions(model, state)
    if not goals:
        return 0.0
    if len(positions) < len(goals):
        return 10000.0
    costs = []
    for x, y in positions:
        if 0 <= x < 64 and 0 <= y < 64:
            costs.append([t[y*64+x] for t in model["distances"]])
        else:
            costs.append([(abs(x-gx)+abs(y-gy))/3.0+10
                          for gx, gy in goals])
    m, g = len(positions), len(goals)
    if g == 1:
        return min(c[0] for c in costs)
    if g == 2 and m == 2:
        return min(costs[0][0]+costs[1][1],
                   costs[0][1]+costs[1][0])
    if m <= 7:
        return min(sum(costs[p[j]][j] for j in range(g))
                   for p in itertools.permutations(range(m), g))
    return sum(min(c[j] for c in costs) for j in range(g))


def move(model, state, ai):
    action = model["actions"][ai]
    s = list(state)
    nb = model["nb"]
    descendants = model["desc"]
    kind = action["kind"]
    for root in action["roots"]:
        k = 4*root
        if kind != "rot":
            delta = 1 if kind == "inc" else -1
            if s[k+2]+delta < 1:
                continue
            s[k+2] += delta
            dx, dy = DIRS[s[k+3]]
            dx, dy = 3*delta*dx, 3*delta*dy
            for child in descendants[root]:
                j = 4*child
                s[j] += dx
                s[j+1] += dy
        else:
            parent = model["parents"][root]
            turns = 1
            if parent is not None and parent < nb:
                if abs(s[k+3]-1-s[4*parent+3]) == 2:
                    turns = 2
            for _ in range(turns):
                px, py = s[k], s[k+1]
                for child in descendants[root]:
                    j = 4*child
                    x, y, a, b = s[j:j+4]
                    if child < nb:
                        s[j] = px+y-py
                        s[j+1] = py-x+px
                        s[j+3] = (b-1) % 4
                    else:
                        s[j] = px+y-py
                        s[j+1] = py-x+px-(a-3)
                        s[j+2], s[j+3] = b, a
                s[k+3] = (s[k+3]-1) % 4
    return tuple(s)


def validate(model, state, old_boxes, ai):
    affected = model["actions"][ai]["affected"]
    boxes = list(old_boxes)
    for i in affected:
        boxes[i] = rectangle(state, i, model["nb"])
        if wall_count(model, boxes[i], i):
            return None
    changed = set(affected)
    for i in affected:
        for j, box in enumerate(boxes):
            if i == j or (j in changed and j < i):
                continue
            if overlap(boxes[i], box):
                return None
    return tuple(boxes)


def render_model(model, state):
    frame = np.full((64, 64), 5, dtype=np.int8)
    frame[model["static"]] = 15
    nb = model["nb"]

    def paint(x, y, w, h, color):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(64, x+w), min(64, y+h)
        if x0 < x1 and y0 < y1:
            frame[y0:y1, x0:x1] = color

    for i in range(nb, model["n"]):
        paint(*rectangle(state, i, nb), 1)
    for i, color in enumerate(model["colors"]):
        x, y, w, h = rectangle(state, i, nb)
        paint(x, y, w, h, color)
        direction = state[4*i+3]
        if direction == 0:
            paint(x, y+h-1, w, 1, 3)
        elif direction == 1:
            paint(x, y, 1, h, 3)
        elif direction == 2:
            paint(x, y, w, 1, 3)
        else:
            paint(x+w-1, y, 1, h, 3)
    for x, y in marker_positions(model, state):
        paint(x+1, y+1, 1, 1, 13)
    for x, y in model["goals"]:
        for xx, yy in ((x+1, y), (x, y+1),
                       (x+2, y+1), (x+1, y+2)):
            paint(xx, yy, 1, 1, 13)
    frame[model["ui"]] = model["ui_pixels"][model["ui"]]
    return frame


def matches(model, state, obs):
    return np.array_equal(render_model(model, state)[:63], obs[:63])


def associate(model, fresh, fresh_state, before, predicted):
    if model["nb"] != fresh["nb"] or model["n"] != fresh["n"]:
        return None
    nb, n = model["nb"], model["n"]
    mapping = {}

    def assign(old_ids, new_ids):
        if len(old_ids) != len(new_ids):
            return False

        def cost(i, j):
            v = fresh_state[4*j:4*j+4]
            scores = []
            for st in (before, predicted):
                u = st[4*i:4*i+4]
                score = abs(u[0]-v[0])+abs(u[1]-v[1])
                if i < nb:
                    score += 4*abs(u[2]-v[2])+8*(u[3] != v[3])
                else:
                    score += 2*(abs(u[2]-v[2])+abs(u[3]-v[3]))
                scores.append(score)
            return min(scores)

        if len(old_ids) <= 7:
            permutation = min(
                itertools.permutations(new_ids),
                key=lambda pp: sum(cost(i, j)
                                   for i, j in zip(old_ids, pp)))
            mapping.update(zip(old_ids, permutation))
        else:
            remaining = set(new_ids)
            for i in old_ids:
                j = min(remaining, key=lambda jj: cost(i, jj))
                mapping[i] = j
                remaining.remove(j)
        return True

    for color in set(model["colors"]):
        old_ids = [i for i, c in enumerate(model["colors"]) if c == color]
        new_ids = [i for i, c in enumerate(fresh["colors"]) if c == color]
        if not assign(old_ids, new_ids):
            return None
    if not assign(list(range(nb, n)), list(range(nb, n))):
        return None
    actual = []
    for i in range(n):
        j = mapping[i]
        actual.extend(fresh_state[4*j:4*j+4])
    return tuple(actual)


def learn(model, before, predicted, actual, ai, obs):
    action = model["actions"][ai]
    roots = set(action["roots"])
    nb, n = model["nb"], model["n"]
    changed = {i for i in range(n)
               if actual[4*i:4*i+4] != before[4*i:4*i+4]}
    for i in range(n):
        if i in roots:
            continue
        old = before[4*i:4*i+4]
        pred = predicted[4*i:4*i+4]
        now = actual[4*i:4*i+4]
        if pred != old and now == old:
            model["parents"][i] = None
    update_closures(model)

    for i in sorted(changed-roots):
        parent = model["parents"][i]
        if parent in changed:
            continue
        if any(i in model["desc"][r] for r in roots):
            continue
        old = before[4*i:4*i+4]
        now = actual[4*i:4*i+4]
        possible = []
        for r in roots:
            k = 4*r
            if action["kind"] != "rot":
                delta = actual[k+2]-before[k+2]
                dx, dy = DIRS[before[k+3]]
                candidate = (old[0]+3*delta*dx, old[1]+3*delta*dy,
                             old[2], old[3])
                if delta and candidate == now:
                    possible.append(r)
            else:
                px, py = before[k:k+2]
                v = old
                for _ in range(2):
                    x, y, a, b = v
                    if i < nb:
                        v = (px+y-py, py-x+px, a, (b-1) % 4)
                    else:
                        v = (px+y-py, py-x+px-(a-3), b, a)
                    if v == now:
                        possible.append(r)
                        break
        if possible:
            r = min(possible, key=lambda rr:
                    abs(before[4*rr]-old[0])+abs(before[4*rr+1]-old[1]))
            if r not in model["desc"][i]:
                model["parents"][i] = r
                update_closures(model)

    _, marks = find_marks(obs, model["ui"])
    bind_markers(model, actual, marks)
    update_capabilities(model)
    distance_tables(model, actual)
    model.pop("ik_goals", None)


def features(model, state):
    nb = model["nb"]
    lengths = tuple(state[4*i+2] for i in range(nb))
    relative = []
    for i in range(nb):
        parent = model["parents"][i]
        direction = state[4*i+3]
        if parent is not None and parent < nb:
            direction = (direction-state[4*parent+3]) % 4
        relative.append(direction)
    return lengths, tuple(relative)


def turn_distance(old, new, child):
    if old == new:
        return 0
    direction = old
    for steps in range(1, 5):
        direction = (direction-1) % 4
        if child and direction == 2:
            direction = (direction-1) % 4
        if direction == new:
            return steps
    return 30


def feature_distance(model, f, goal_f):
    lengths, relative = f
    goal_lengths, goal_relative = goal_f
    length_cost = {}
    turn_cost = {}
    for i, color in enumerate(model["colors"]):
        delta = abs(lengths[i]-goal_lengths[i])
        if delta and i not in model["resizable"]:
            return 10000
        length_cost[color] = max(length_cost.get(color, 0), delta)
        if relative[i] != goal_relative[i]:
            if i not in model["rotatable"]:
                return 10000
            parent = model["parents"][i]
            child = parent is not None and parent < model["nb"]
            distance = turn_distance(relative[i], goal_relative[i], child)
            turn_cost[color] = max(turn_cost.get(color, 0), distance)
    return sum(length_cost.values())+sum(turn_cost.values())


def generate_ik_goals(model, start, deadline):
    nb = model["nb"]
    if not model["rotatable"]:
        return []
    if any(len(a["roots"]) > 1 for a in model["actions"]
           if a["kind"] == "rot"):
        return []

    positions = marker_positions(model, start)
    occupied = set(positions)
    uncovered = [g for g in model["goals"] if g not in occupied]
    if not uncovered:
        return []

    movable_markers = [
        (mi, owner, offsets)
        for mi, (owner, offsets) in enumerate(model["markers"])
        if positions[mi] not in model["goals"]]
    if not movable_markers:
        return []

    found = {}
    start_f = features(model, start)
    node_count = 0
    rotatable = model["rotatable"]
    resizable = model["resizable"]

    for mi, owner, offsets in movable_markers:
        chain = [owner]
        seen = {owner}
        while model["parents"][chain[-1]] is not None:
            parent = model["parents"][chain[-1]]
            if parent >= nb or parent in seen:
                break
            chain.append(parent)
            seen.add(parent)
        chain.reverse()
        root = chain[0]
        if model["parents"][root] is not None:
            continue
        if not (2 <= len(chain) <= 5):
            continue
        if set(model["desc"][root]) != set(chain[1:]):
            continue
        if not any(i in rotatable for i in chain):
            continue
        if any(len(a["roots"]) != 1
               for a in model["actions"]
               if any(i in chain for i in a["roots"])):
            continue

        valid_geometry = True
        for parent, child in zip(chain, chain[1:]):
            px, py, pn, pd = start[4*parent:4*parent+4]
            dx, dy = DIRS[pd]
            if start[4*child:4*child+2] != (px+3*pn*dx, py+3*pn*dy):
                valid_geometry = False
        if not valid_geometry:
            continue

        chain_set = set(chain)
        marker_owners = {i for i, _ in model["markers"]}
        auxiliary, locked = [], []
        for i in range(model["n"]):
            if i in chain_set:
                continue
            if (i < nb and model["parents"][i] is None
                    and not model["desc"][i] and i not in marker_owners):
                auxiliary.append(i)
            else:
                locked.append(rectangle(start, i, nb))

        relative = {
            child: (start[4*child+3]-start[4*parent+3]) % 4
            for parent, child in zip(chain, chain[1:])}
        state = list(start)

        def save_candidate(completed):
            candidate = tuple(completed)
            points = set(marker_positions(model, candidate))
            if not occupied.intersection(model["goals"]).issubset(points):
                return
            cf = features(model, candidate)
            cost = feature_distance(model, start_f, cf)
            if cost >= 10000:
                return
            signature = cf[1]
            bucket = found.setdefault(signature, [])
            if candidate not in {s for _, s in bucket}:
                bucket.append((cost, candidate))
                bucket.sort(key=lambda z: z[0])
                del bucket[3:]

        def finish(boxes):
            completed = list(state)

            def place_auxiliary(index, all_boxes):
                if index == len(auxiliary):
                    save_candidate(completed)
                    return
                i = auxiliary[index]
                k = 4*i
                completed[k:k+4] = start[k:k+4]
                directions = range(4) if i in rotatable else (start[k+3],)
                lengths = (range(start[k+2], 0, -1)
                           if i in resizable else (start[k+2],))
                for direction in directions:
                    completed[k+3] = direction
                    for length in lengths:
                        completed[k+2] = length
                        r = rectangle(completed, i, nb)
                        if wall_count(model, r, i):
                            continue
                        if any(overlap(r, b) for b in all_boxes):
                            continue
                        place_auxiliary(index+1, all_boxes+(r,))
                        # Closest feasible length for this orientation suffices.
                        break

            place_auxiliary(0, tuple(boxes)+tuple(locked))

        def dfs(depth, ax, ay, parent_direction, boxes, target):
            nonlocal node_count
            node_count += 1
            if node_count % 128 == 0 and time.monotonic() >= deadline:
                return
            i = chain[depth]
            k = 4*i
            if i in rotatable:
                directions = range(4)
            elif depth == 0:
                directions = (start[k+3],)
            else:
                directions = ((parent_direction+relative[i]) % 4,)

            for direction in directions:
                if depth and i in rotatable:
                    if (direction-parent_direction) % 4 == 2:
                        continue
                state[k], state[k+1], state[k+3] = ax, ay, direction
                dx, dy = DIRS[direction]
                if depth == len(chain)-1:
                    ox, oy = offsets[direction]
                    ex, ey = target[0]-ax-ox, target[1]-ay-oy
                    if dx == 0:
                        if ex != 0 or ey*dy < 0 or ey % 3:
                            continue
                        length = ey*dy//3+1
                    else:
                        if ey != 0 or ex*dx < 0 or ex % 3:
                            continue
                        length = ex*dx//3+1
                    if not (1 <= length <= 21):
                        continue
                    if i not in resizable and length != start[k+2]:
                        continue
                    lengths = (length,)
                else:
                    lengths = (range(1, 22)
                               if i in resizable else (start[k+2],))

                for length in lengths:
                    state[k+2] = length
                    r = rectangle(state, i, nb)
                    if wall_count(model, r, i):
                        if depth != len(chain)-1 and i in resizable:
                            break
                        continue
                    if any(overlap(r, b) for b in boxes):
                        continue
                    if any(overlap(r, b) for b in locked):
                        continue
                    new_boxes = boxes+(r,)
                    if depth == len(chain)-1:
                        finish(new_boxes)
                    else:
                        dfs(depth+1, ax+3*length*dx, ay+3*length*dy,
                            direction, new_boxes, target)
                    if time.monotonic() >= deadline:
                        return

        for target in uncovered:
            dfs(0, start[4*root], start[4*root+1], None, (), target)
            if time.monotonic() >= deadline:
                break
        if time.monotonic() >= deadline:
            break

    entries = [entry for bucket in found.values() for entry in bucket]
    entries.sort(key=lambda e: e[0])
    return [s for _, s in entries[:20]]


def reconstruct(parent, end, start):
    path = []
    while end != start:
        entry = parent.get(end)
        if entry is None:
            return []
        end, ai = entry
        path.append(ai)
    path.reverse()
    return path


def clearance_plan(model, start, goals, bans, deadline):
    """Prepare independent obstacles, allowing temporary clearance changes."""
    if not goals:
        return []
    marked = {i for i, _ in model["markers"]}
    independent = {
        i for i in range(model["nb"])
        if model["parents"][i] is None and not model["desc"][i]}
    auxiliary = sorted(independent-marked)
    if not auxiliary:
        return []
    allowed = [
        ai for ai, a in enumerate(model["actions"])
        if all(i in independent for i in a["roots"])]
    if not allowed:
        return []

    # Try a few diverse auxiliary arrangements, with nearer ones first.
    signatures = []
    for goal in goals:
        signature = tuple((i, goal[4*i+2], goal[4*i+3])
                          for i in auxiliary)
        if signature not in signatures:
            signatures.append(signature)
    if any(all(start[4*i+2:4*i+4] == (n, d)
               for i, n, d in signature)
           for signature in signatures):
        return []
    signatures = signatures[:6]
    preserve = tuple(sorted(independent & marked))

    def target(state):
        if any(state[4*i:4*i+4] != start[4*i:4*i+4] for i in preserve):
            return False
        return any(all(state[4*i+2:4*i+4] == (n, d)
                       for i, n, d in signature)
                   for signature in signatures)

    initial_boxes = tuple(rectangle(start, i, model["nb"])
                          for i in range(model["n"]))
    queue = collections.deque([(start, initial_boxes, 0)])
    seen = {start}
    parent = {}
    count = 0
    while queue:
        state, boxes, depth = queue.popleft()
        count += 1
        if count % 64 == 0 and time.monotonic() >= deadline:
            break
        if target(state):
            return reconstruct(parent, state, start)
        if depth >= 18:
            continue
        for ai in allowed:
            if (state, ai) in bans:
                continue
            new = move(model, state, ai)
            if new in seen or new == state:
                continue
            new_boxes = validate(model, new, boxes, ai)
            if new_boxes is None:
                continue
            seen.add(new)
            parent[new] = (state, ai)
            queue.append((new, new_boxes, depth+1))
        if len(seen) > 16000:
            break
    return []


def search(model, start, bans, visits, deadline):
    if won(model, start):
        return []
    actions = model["actions"]
    if not actions or not model["markers"]:
        return []

    ik = model.get("ik_goals")
    if ik and start in ik:
        model.pop("ik_goals", None)
        ik = None
    if ik is None:
        generation_deadline = min(deadline-1.0, time.monotonic()+1.65)
        ik = generate_ik_goals(model, start, generation_deadline)
        model["ik_goals"] = ik

    preparation = clearance_plan(
        model, start, ik, bans,
        min(deadline-1.0, time.monotonic()+0.65))
    if preparation:
        return preparation

    goal_set = set(ik)
    goal_features = [features(model, s) for s in ik]

    def heuristic(state):
        mh = marker_heuristic(model, state)
        if not goal_features:
            return mh
        f = features(model, state)
        kh = min(feature_distance(model, f, gf) for gf in goal_features)
        return kh+0.12*mh

    boxes = tuple(rectangle(start, i, model["nb"])
                  for i in range(model["n"]))
    h0 = heuristic(start)
    weight = 1.65 if goal_features else 2.35
    heap = [(weight*h0, h0, 0, 0, start, boxes)]
    best_g = {start: 0}
    parent = {}
    counter = 0
    expanded = 0
    best_state = None
    best_score = float("inf")
    solution = None

    while heap:
        if expanded % 64 == 0 and time.monotonic() >= deadline:
            break
        _, h, g, _, state, boxes = heapq.heappop(heap)
        if best_g.get(state) != g:
            continue
        expanded += 1
        if state != start:
            score = h+0.025*g+0.6*visits.get(state, 0)
            if score < best_score:
                best_score, best_state = score, state
        if won(model, state) or state in goal_set:
            solution = state
            break

        for ai in range(len(actions)):
            if (state, ai) in bans:
                continue
            new = move(model, state, ai)
            if new == state:
                continue
            ng = g+1
            if new != start:
                ng += min(2, visits.get(new, 0))
            if best_g.get(new, 10**9) <= ng:
                continue
            new_boxes = validate(model, new, boxes, ai)
            if new_boxes is None:
                continue
            nh = heuristic(new)
            best_g[new] = ng
            parent[new] = (state, ai)
            counter += 1
            heapq.heappush(heap, (ng+weight*nh, nh, ng, counter,
                                  new, new_boxes))
        if expanded >= 180000 or len(best_g) >= 450000:
            break

    end = solution if solution is not None else best_state
    if end is None:
        return []
    path = reconstruct(parent, end, start)
    if solution is None:
        return path[:min(7, len(path))]
    return path


def act(obs, available, memory):
    started = time.monotonic()
    obs = np.asarray(obs, dtype=np.int8)
    cache = memory.get("_player")
    model = state = None

    if cache is not None:
        model, state = cache["model"], cache["state"]
        if not matches(model, state, obs):
            before, ai = cache.get("before"), cache.get("last_action")
            cache["plan"] = []
            if before is not None and matches(model, before, obs):
                if ai is not None:
                    cache["bans"].add((before, ai))
                state = before
                cache["state"] = state
            else:
                fresh, fresh_state = make_model(obs)
                actual = None
                if fresh is not None and before is not None and ai is not None:
                    actual = associate(model, fresh, fresh_state, before, state)
                if actual is not None:
                    learn(model, before, state, actual, ai, obs)
                    state = actual
                    cache["state"] = state
                    if not matches(model, state, obs):
                        model = None
                else:
                    model = None
                if model is None:
                    model, state = fresh, fresh_state
                    cache = None

    if model is None:
        model, state = make_model(obs)
        cache = None

    if model is None:
        controls, _ = controls_and_ui(obs)
        if controls:
            _, _, row, col = controls[0]
            return "ACTION6", int(row), int(col)
        if "RESET" in available:
            return "RESET", None, None
        return "ACTION6", 0, 0

    if cache is None:
        cache = dict(model=model, state=state, plan=[],
                     bans=set(), visits={})
        memory["_player"] = cache
    cache["model"], cache["state"] = model, state
    visits = cache["visits"]
    visits[state] = visits.get(state, 0)+1

    if won(model, state):
        return "ACTION6", 0, 0

    if not cache["plan"]:
        cache["plan"] = search(model, state, cache["bans"], visits,
                               started+8.55)

    if cache["plan"]:
        ai = cache["plan"].pop(0)
    else:
        boxes = tuple(rectangle(state, i, model["nb"])
                      for i in range(model["n"]))
        candidates = []
        goal_features = [
            features(model, goal) for goal in model.get("ik_goals", [])]
        for j in range(len(model["actions"])):
            if (state, j) in cache["bans"]:
                continue
            new = move(model, state, j)
            if new == state or validate(model, new, boxes, j) is None:
                continue
            score = marker_heuristic(model, new)
            if goal_features:
                nf = features(model, new)
                score = (min(feature_distance(model, nf, f)
                             for f in goal_features)+0.12*score)
            score += 4*visits.get(new, 0)
            candidates.append((score, j))
        if not candidates:
            if "RESET" in available and visits[state] > 2:
                memory.pop("_player", None)
                return "RESET", None, None
            return "ACTION6", 0, 0
        ai = min(candidates)[1]

    action = model["actions"][ai]
    cache["before"] = state
    cache["last_action"] = ai
    cache["state"] = move(model, state, ai)
    return "ACTION6", int(action["row"]), int(action["col"])

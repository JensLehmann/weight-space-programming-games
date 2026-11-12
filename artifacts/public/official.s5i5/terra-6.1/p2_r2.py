import heapq
import itertools
from collections import Counter

import numpy as np


DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _buttons(obs):
    buttons = []
    mask = np.zeros((64, 64), dtype=bool)
    sizes = ((13, 7), (7, 13), (11, 5), (5, 11), (7, 7), (5, 5))

    for y in range(63):
        for x in range(64):
            if int(obs[y, x]) != 2 or mask[y, x]:
                continue
            for w, h in sizes:
                if x + w > 64 or y + h > 63:
                    continue
                a = obs[y:y + h, x:x + w]
                if not (np.all(a[0] == 2) and np.all(a[-1] == 2)
                        and np.all(a[:, 0] == 2)
                        and np.all(a[:, -1] == 2)):
                    continue
                inside = a[1:-1, 1:-1]
                if not np.any(inside == 4):
                    continue
                colors = [int(v) for v in inside.flat
                          if int(v) not in (2, 3, 4, 5, 13)]
                if not colors:
                    continue
                color = Counter(colors).most_common(1)[0][0]
                if np.any(inside == 3):
                    if w > h:
                        buttons.append((color, -1, (y + h // 2, x + 1)))
                        buttons.append((color, 1, (y + h // 2, x + w - 2)))
                    else:
                        buttons.append((color, -1, (y + 1, x + w // 2)))
                        buttons.append((color, 1, (y + h - 2, x + w // 2)))
                else:
                    buttons.append((color, 0, (y + h // 2, x + w // 2)))
                mask[y:y + h, x:x + w] = True
                break
    return buttons, mask


def _components(cells):
    cells = set(cells)
    result = []
    while cells:
        p = cells.pop()
        stack = [p]
        component = [p]
        while stack:
            x, y = stack.pop()
            for q in ((x + 1, y), (x - 1, y),
                      (x, y + 1), (x, y - 1)):
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
                    component.append(q)
        result.append(component)
    return result


def _scene(obs):
    buttons, ui = _buttons(obs)
    if not buttons:
        return None

    tile_color = {}
    anchors = {}
    dots = []
    goals = []
    excluded = {2, 3, 4, 5, 13}

    for ty in range(21):
        for tx in range(21):
            y, x = 3 * ty, 3 * tx
            a = obs[y:y + 3, x:x + 3]
            m = ui[y:y + 3, x:x + 3]

            if not m[1, 1] and int(a[1, 1]) == 13:
                dots.append((tx, ty))
            edge = ((0, 1), (1, 0), (1, 2), (2, 1))
            if all(not m[dy, dx] and int(a[dy, dx]) == 13
                   for dy, dx in edge):
                goals.append((tx, ty))

            counts = Counter(
                int(a[dy, dx])
                for dy in range(3)
                for dx in range(3)
                if not m[dy, dx] and int(a[dy, dx]) not in excluded
            )
            if not counts:
                continue
            color, count = counts.most_common(1)[0]
            if count < 2:
                continue
            tile_color[(tx, ty)] = color

            if color in (1, 15):
                continue
            sides = (
                sum(not m[0, k] and int(a[0, k]) == 3 for k in range(3)),
                sum(not m[k, 0] and int(a[k, 0]) == 3 for k in range(3)),
                sum(not m[2, k] and int(a[2, k]) == 3 for k in range(3)),
                sum(not m[k, 2] and int(a[k, 2]) == 3 for k in range(3)),
            )
            side = max(range(4), key=lambda i: sides[i])
            if sides[side] >= 2:
                # Directions are right, down, left, up.
                anchors[(tx, ty)] = (color, (1, 0, 3, 2)[side])

    if not dots or not goals or len(dots) < len(goals):
        return None

    nodes = []
    state = []
    owner = {}
    occupied = set()
    rod_anchors = []

    for (ax, ay), (color, direction) in sorted(anchors.items()):
        dx, dy = DIRS[direction]
        length = 1
        while length < 21:
            p = (ax + dx * length, ay + dy * length)
            if tile_color.get(p) != color or p in anchors:
                break
            length += 1

        x = ax + min(0, dx * (length - 1))
        y = ay + min(0, dy * (length - 1))
        index = len(nodes)
        nodes.append(("rod", color, None))
        state.extend((x, y, direction, length))
        rod_anchors.append((ax, ay))

        for k in range(length):
            p = (ax + dx * k, ay + dy * k)
            owner[p] = index
            occupied.add(p)

    rod_count = len(nodes)
    if not rod_count:
        return None

    remaining = {}
    for p, color in tile_color.items():
        if p not in occupied:
            remaining.setdefault(color, []).append(p)

    walls = set(remaining.pop(15, []))
    for color, cells in remaining.items():
        for component in _components(cells):
            ox = min(x for x, y in component)
            oy = min(y for x, y in component)
            offsets = [(x - ox, y - oy) for x, y in component]
            rotations = []
            current = offsets
            for _ in range(4):
                rotations.append(tuple(current))
                width = max(x for x, y in current) + 1
                current = [(y, width - 1 - x) for x, y in current]
            index = len(nodes)
            nodes.append(("shape", color, tuple(rotations)))
            state.extend((ox, oy, 0, 0))
            for p in component:
                owner[p] = index

    children = [[] for _ in nodes]
    dot_owners = {owner[p] for p in dots if p in owner}

    def cycle(parent, child):
        stack = [child]
        visited = set()
        while stack:
            i = stack.pop()
            if i == parent:
                return True
            if i in visited:
                continue
            visited.add(i)
            stack.extend(children[i])
        return False

    # The free end of a rod commonly connects to the next piece.
    proposals = []
    for i in range(rod_count):
        ax, ay = rod_anchors[i]
        direction = state[4 * i + 2]
        length = state[4 * i + 3]
        dx, dy = DIRS[direction]
        p = (ax + dx * length, ay + dy * length)
        j = owner.get(p)
        if j is None or j == i:
            continue
        if nodes[j][0] == "rod":
            if p != rod_anchors[j]:
                continue
            # For two single-cell pieces, a marked piece can carry
            # an adjacent unmarked piece.
            if (length == 1 and state[4 * j + 3] == 1
                    and j in dot_owners and i not in dot_owners):
                proposals.append((j, i))
            else:
                proposals.append((i, j))
        elif nodes[j][1] == 1:
            proposals.append((i, j))

    for parent, child in proposals:
        if not cycle(parent, child) and child not in children[parent]:
            children[parent].append(child)

    dot_indices = []
    for x, y in dots:
        index = len(nodes)
        nodes.append(("dot", 13, None))
        state.extend((x, y, 0, 0))
        children.append([])
        dot_indices.append(index)
        parent = owner.get((x, y))
        if parent is not None:
            children[parent].append(index)

    parents = [[] for _ in nodes]
    for i, cs in enumerate(children):
        for j in cs:
            parents[j].append(i)

    operations = []
    seen = set()
    rod_colors = {nodes[i][1] for i in range(rod_count)}
    for color, kind, click in buttons:
        if color not in rod_colors or (color, kind) in seen:
            continue
        seen.add((color, kind))
        affected = tuple(i for i in range(rod_count)
                         if nodes[i][1] == color)
        operations.append((color, kind, click, affected))

    if not operations:
        return None

    return {
        "nodes": nodes,
        "state": tuple(state),
        "children": children,
        "parents": parents,
        "walls": walls,
        "dots": dot_indices,
        "goals": goals,
        "ops": operations,
    }


def _search(scene, forbidden, limit=9000):
    nodes = scene["nodes"]
    children = scene["children"]
    parents = scene["parents"]
    walls = scene["walls"]
    dot_indices = scene["dots"]
    goals = scene["goals"]
    operations = scene["ops"]
    start = scene["state"]

    assignments = list(itertools.islice(
        itertools.permutations(range(len(dot_indices)), len(goals)), 5000
    ))

    def heuristic(s):
        best = 10000
        for assignment in assignments:
            total = 0
            for (gx, gy), k in zip(goals, assignment):
                j = 4 * dot_indices[k]
                total += abs(s[j] - gx) + abs(s[j + 1] - gy)
            if total < best:
                best = total
        return best

    def translate(s, i, dx, dy):
        j = 4 * i
        s[j] += dx
        s[j + 1] += dy
        for child in children[i]:
            translate(s, child, dx, dy)

    def child_rotate(s, i, px, py):
        j = 4 * i
        x, y, d, length = s[j:j + 4]
        typ = nodes[i][0]
        if typ == "rod":
            width = length if d in (0, 2) else 1
        elif typ == "shape":
            width = max(a for a, b in nodes[i][2][d]) + 1
        else:
            width = 1
        s[j] = px - (y - py)
        s[j + 1] = py + (x - px) - (width - 1)
        if typ == "rod":
            s[j + 2] = (d - 1) % 4
        elif typ == "shape":
            s[j + 2] = (d + 1) % 4
        for child in children[i]:
            child_rotate(s, child, px, py)

    def root_rotate(s, i):
        j = 4 * i
        x, y, d, length = s[j:j + 4]
        if d == 0:
            px, py = x, y
        elif d == 1:
            px, py = x, y
        elif d == 2:
            px, py = x + length - 1, y
        else:
            px, py = x, y + length - 1

        for child in children[i]:
            child_rotate(s, child, px, py)

        if d == 0:
            s[j + 1] -= length - 1
        elif d == 2:
            s[j] += length - 1
        elif d == 3:
            s[j] -= length - 1
            s[j + 1] += length - 1
        s[j + 2] = (d - 1) % 4

    def clear(s):
        used = set(walls)
        for i, node in enumerate(nodes):
            typ = node[0]
            if typ == "dot":
                continue
            j = 4 * i
            x, y, d, length = s[j:j + 4]
            if typ == "rod":
                if d in (0, 2):
                    cells = ((x + k, y) for k in range(length))
                else:
                    cells = ((x, y + k) for k in range(length))
            else:
                cells = ((x + dx, y + dy) for dx, dy in node[2][d])
            for p in cells:
                if not (0 <= p[0] < 21 and 0 <= p[1] < 21):
                    return False
                if p in used:
                    return False
                used.add(p)
        return True

    def transition(s, op):
        _, kind, _, affected = op
        t = list(s)
        changed = False

        if kind:
            for i in affected:
                j = 4 * i
                d, length = t[j + 2], t[j + 3]
                new_length = length + kind
                if new_length < 1 or new_length > 21:
                    continue
                dx, dy = DIRS[d]
                if d == 2:
                    t[j] -= kind
                elif d == 3:
                    t[j + 1] -= kind
                t[j + 3] = new_length
                for child in children[i]:
                    translate(t, child, dx * kind, dy * kind)
                changed = True
        else:
            for i in affected:
                j = 4 * i
                turns = 1
                if parents[i]:
                    p = parents[i][0]
                    if nodes[p][0] == "rod":
                        if (t[j + 2] - t[4 * p + 2]) % 4 == 3:
                            turns = 2
                for _ in range(turns):
                    root_rotate(t, i)
                changed = True

        if not changed:
            return None
        result = tuple(t)
        if result == s or not clear(result):
            return None
        return result

    h0 = heuristic(start)
    if h0 == 0:
        return []

    serial = itertools.count()
    queue = [(1.8 * h0, h0, 0, next(serial), start)]
    distance = {start: 0}
    previous = {}
    best_state = None
    best_key = (h0, float("inf"))
    expansions = 0

    while queue and expansions < limit:
        _, h, g, _, s = heapq.heappop(queue)
        if distance.get(s) != g:
            continue
        expansions += 1

        if s != start and (h, g) < best_key:
            best_key = (h, g)
            best_state = s
        if h == 0:
            best_state = s
            break

        for op_index, op in enumerate(operations):
            if s == start and op[2] in forbidden:
                continue
            nxt = transition(s, op)
            if nxt is None:
                continue
            ng = g + 1
            if ng >= distance.get(nxt, 10**9):
                continue
            distance[nxt] = ng
            previous[nxt] = (s, op_index)
            nh = heuristic(nxt)
            heapq.heappush(queue, (
                ng + 1.8 * nh, nh, ng, next(serial), nxt
            ))

    if best_state is None:
        return None

    plan = []
    while best_state != start:
        parent, op_index = previous[best_state]
        positions = [[best_state[4 * i], best_state[4 * i + 1]]
                     for i in dot_indices]
        row, col = operations[op_index][2]
        plan.append([row, col, positions])
        best_state = parent
    plan.reverse()
    return plan


def _signature(obs):
    a = obs[:63].astype(np.int64).reshape(-1)
    i = np.arange(1, len(a) + 1, dtype=np.int64)
    s1 = int(np.sum(a * i))
    s2 = int(np.sum(a * i * i))
    return str(s1) + ":" + str(s2)


def act(obs, available, memory):
    obs = np.asarray(obs)
    if "ACTION6" not in available:
        if "RESET" in available:
            return ("RESET", None, None)
        return (available[0], None, None)

    signature = _signature(obs)
    tried = memory.setdefault("tried", {})
    failed = memory.setdefault("failed", {})

    old_signature = memory.get("last_signature")
    old_click = memory.get("last_click")
    if old_signature == signature and old_click is not None:
        bad = failed.setdefault(signature, [])
        if old_click not in bad:
            bad.append(old_click)
        memory["plan"] = []
    else:
        expected = memory.get("expected")
        if expected:
            for x, y in expected:
                if 0 <= x < 21 and 0 <= y < 21:
                    if int(obs[3 * y + 1, 3 * x + 1]) != 13:
                        memory["plan"] = []
                        break

    plan = memory.get("plan", [])
    if not plan:
        scene = _scene(obs)
        if scene is None:
            if "RESET" in available:
                return ("RESET", None, None)
            return ("ACTION6", 0, 0)

        forbidden = {tuple(p) for p in failed.get(signature, [])}
        forbidden.update(tuple(p) for p in tried.get(signature, []))
        plan = _search(scene, forbidden)

        if not plan:
            forbidden = {tuple(p) for p in failed.get(signature, [])}
            plan = _search(scene, forbidden, limit=3000)

        if not plan:
            choices = [op[2] for op in scene["ops"]
                       if op[2] not in forbidden]
            if choices:
                row, col = choices[0]
                plan = [[row, col, []]]
            elif "RESET" in available:
                memory["plan"] = []
                return ("RESET", None, None)
            else:
                row, col = scene["ops"][0][2]
                plan = [[row, col, []]]

    row, col, expected = plan.pop(0)
    memory["plan"] = plan
    memory["expected"] = expected
    memory["last_signature"] = signature
    memory["last_click"] = [int(row), int(col)]

    used = tried.setdefault(signature, [])
    click = [int(row), int(col)]
    if click not in used:
        used.append(click)

    if len(tried) > 200:
        tried.pop(next(iter(tried)))
    if len(failed) > 100:
        failed.pop(next(iter(failed)))

    return ("ACTION6", int(row), int(col))

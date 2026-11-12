import heapq
import itertools
import time
from collections import Counter, deque
from functools import lru_cache

DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
COLORS = (6, 7, 8, 9, 10, 11, 12, 14)
REL_INDEX = {0: 0, 3: 1, 1: 2}


def controls_from_frame(obs):
    controls = []
    covered = [[False] * 64 for _ in range(64)]

    for y in range(63):
        x = 0
        while x < 64:
            if int(obs[y, x]) != 2 or (x and int(obs[y, x - 1]) == 2):
                x += 1
                continue
            end = x
            while end < 64 and int(obs[y, end]) == 2:
                end += 1
            w = end - x

            if 5 <= w <= 19:
                for h in range(5, min(20, 64 - y) + 1):
                    if any(int(obs[y + h - 1, xx]) != 2
                           for xx in range(x, end)):
                        continue
                    if any(int(obs[yy, x]) != 2 or
                           int(obs[yy, end - 1]) != 2
                           for yy in range(y, y + h)):
                        continue

                    counts = Counter(
                        int(obs[yy, xx])
                        for yy in range(y + 1, y + h - 1)
                        for xx in range(x + 1, end - 1)
                        if int(obs[yy, xx]) not in (2, 3, 4, 5, 13, 15)
                    )
                    if not counts:
                        continue
                    color = counts.most_common(1)[0][0]

                    if w == h and w <= 7:
                        kind = "rotate"
                        clicks = [(y + h // 2, x + w // 2)]
                    elif w > h:
                        kind = "resize"
                        clicks = [
                            (y + h // 2, x + 1),
                            (y + h // 2, end - 2),
                        ]
                    elif h > w:
                        kind = "resize"
                        clicks = [
                            (y + 1, x + w // 2),
                            (y + h - 2, x + w // 2),
                        ]
                    else:
                        continue

                    controls.append((color, kind, clicks))
                    for yy in range(y, y + h):
                        for xx in range(x, end):
                            covered[yy][xx] = True
                    break
            x = end
    return controls, covered


def components(cells):
    cells = set(cells)
    result = []
    while cells:
        p = cells.pop()
        stack = [p]
        comp = [p]
        while stack:
            x, y = stack.pop()
            for dx, dy in DIRS:
                q = (x + dx, y + dy)
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
                    comp.append(q)
        result.append(comp)
    return result


def parse_frame(obs):
    controls, covered = controls_from_frame(obs)
    controlled = {c for c, _, _ in controls}
    object_colors = set(COLORS) | {
        c for c in controlled if c not in (1, 15)
    }

    cell_color = {}
    anchors = {}
    dots = []
    goals = []

    for cy in range(21):
        for cx in range(21):
            vals = [
                None if covered[3 * cy + yy][3 * cx + xx]
                else int(obs[3 * cy + yy, 3 * cx + xx])
                for yy in range(3) for xx in range(3)
            ]
            if vals[4] == 13:
                dots.append((cx, cy))
            if all(vals[i] == 13 for i in (1, 3, 5, 7)):
                goals.append((cx, cy))

            count = Counter(v for v in vals if v is not None)
            possible = [
                c for c in object_colors | {1, 15}
                if count[c] >= 2
            ]
            if not possible:
                continue
            color = max(possible, key=lambda c: count[c])
            cell_color[(cx, cy)] = color

            if color in object_colors:
                edges = (
                    (6, 7, 8), (0, 3, 6),
                    (0, 1, 2), (2, 5, 8),
                )
                scores = [
                    sum(vals[i] == 3 for i in edge)
                    for edge in edges
                ]
                if max(scores) >= 2:
                    anchors[(cx, cy)] = scores.index(max(scores))

    nodes = []
    used = set()
    for (x, y), r in sorted(
        anchors.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        if (x, y) in used:
            continue
        color = cell_color.get((x, y))
        dx, dy = DIRS[r]
        length = 0
        while True:
            p = (x + length * dx, y + length * dy)
            if cell_color.get(p) != color:
                break
            if length and p in anchors:
                break
            if p in used:
                break
            used.add(p)
            length += 1
        if length:
            nodes.append({
                "kind": "bar",
                "color": color,
                "initial": (x, y, r, length),
            })

    remaining = {}
    for p, color in cell_color.items():
        if p not in used:
            remaining.setdefault(color, []).append(p)

    for color, cells in remaining.items():
        for comp in components(cells):
            minx = min(x for x, _ in comp)
            miny = min(y for _, y in comp)
            shape = tuple(sorted((x - minx, y - miny) for x, y in comp))
            shapes = []
            for _ in range(4):
                width = max(x for x, _ in shape) + 1
                height = max(y for _, y in shape) + 1
                shapes.append((shape, width, height))
                shape = tuple(sorted(
                    (y, width - 1 - x) for x, y in shape
                ))
            nodes.append({
                "kind": "fixed",
                "color": color,
                "initial": (minx, miny, 0, 0),
                "shapes": shapes,
            })

    occupancy = {}
    for i, node in enumerate(nodes):
        x, y, r, length = node["initial"]
        if node["kind"] == "bar":
            dx, dy = DIRS[r]
            points = [(x + k * dx, y + k * dy) for k in range(length)]
        else:
            points = [(x + xx, y + yy)
                      for xx, yy in node["shapes"][0][0]]
        for p in points:
            occupancy[p] = i

    parent = [-1] * len(nodes)
    for i, node in enumerate(nodes):
        if node["kind"] != "bar":
            continue
        x, y, r, length = node["initial"]
        dx, dy = DIRS[r]
        neighbor = (x + length * dx, y + length * dy)
        j = occupancy.get(neighbor)
        if j is None or j == i or parent[j] != -1:
            continue
        child = nodes[j]
        if child["kind"] == "bar":
            if child["initial"][:2] != neighbor:
                continue
        elif child["color"] != 1:
            continue
        k = i
        while k != -1 and k != j:
            k = parent[k]
        if k != j:
            parent[j] = i

    children = [[] for _ in nodes]
    for j, i in enumerate(parent):
        if i != -1:
            children[i].append(j)

    dot_owner = [occupancy.get(p, -1) for p in dots]
    descendants = []
    descendant_dots = []
    for i in range(len(nodes)):
        stack = [i]
        seen = set()
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            stack.extend(children[j])
        descendants.append(sorted(seen))
        descendant_dots.append([
            k for k, owner in enumerate(dot_owner) if owner in seen
        ])

    state = []
    for node in nodes:
        state.extend(node["initial"])
    for x, y in dots:
        state.extend((x, y))

    actions = []
    for color, kind, clicks in controls:
        if kind == "resize":
            actions.append((color, "resize", -1, clicks[0]))
            actions.append((color, "resize", 1, clicks[1]))
        else:
            actions.append((color, "rotate", 0, clicks[0]))

    return {
        "nodes": nodes,
        "parent": parent,
        "descendants": descendants,
        "descendant_dots": descendant_dots,
        "dots": dots,
        "dot_owner": dot_owner,
        "goals": goals,
        "actions": actions,
        "state": tuple(state),
    }


def signature(model, state):
    n = len(model["nodes"])
    bars = [
        (node["color"],) + tuple(state[4 * i:4 * i + 4])
        for i, node in enumerate(model["nodes"])
        if node["kind"] == "bar"
    ]
    dots = [
        tuple(state[4 * n + 2 * j:4 * n + 2 * j + 2])
        for j in range(len(model["dots"]))
    ]
    return repr((tuple(sorted(bars)), tuple(sorted(dots))))


def make_physics(model):
    nodes = model["nodes"]
    parent = model["parent"]
    descendants = model["descendants"]
    descendant_dots = model["descendant_dots"]
    dot_offset = 4 * len(nodes)

    bars_by_color = {}
    for i, node in enumerate(nodes):
        if node["kind"] == "bar":
            bars_by_color.setdefault(node["color"], []).append(i)

    @lru_cache(maxsize=120000)
    def node_mask(i, x, y, r, length):
        node = nodes[i]
        if node["kind"] == "bar":
            dx, dy = DIRS[r]
            points = ((x + k * dx, y + k * dy) for k in range(length))
        else:
            points = ((x + xx, y + yy)
                      for xx, yy in node["shapes"][r][0])
        mask = 0
        for xx, yy in points:
            if not (0 <= xx < 21 and 0 <= yy < 21):
                return -1
            mask |= 1 << (21 * yy + xx)
        return mask

    def valid(state):
        occupied = 0
        for i in range(len(nodes)):
            a = 4 * i
            mask = node_mask(i, *state[a:a + 4])
            if mask < 0 or occupied & mask:
                return False
            occupied |= mask
        return True

    def transition(state, action):
        color, kind, amount, _ = action
        selected = bars_by_color.get(color, ())
        if not selected:
            return None
        s = list(state)
        changed = False

        for i in selected:
            a = 4 * i
            x, y, r, length = s[a:a + 4]
            if kind == "resize":
                new_length = length + amount
                if not (1 <= new_length <= 21):
                    continue
                s[a + 3] = new_length
                dx, dy = DIRS[r]
                dx *= amount
                dy *= amount
                for j in descendants[i]:
                    if j != i:
                        s[4 * j] += dx
                        s[4 * j + 1] += dy
                for j in descendant_dots[i]:
                    s[dot_offset + 2 * j] += dx
                    s[dot_offset + 2 * j + 1] += dy
                changed = True
            else:
                turns = 1
                p = parent[i]
                if p != -1 and nodes[p]["kind"] == "bar":
                    if (r - s[4 * p + 2]) % 4 == 3:
                        turns = 2
                for _ in range(turns):
                    px, py = s[a], s[a + 1]
                    for j in descendants[i]:
                        b = 4 * j
                        jx, jy, jr, jl = s[b:b + 4]
                        if nodes[j]["kind"] == "bar":
                            s[b] = px + jy - py
                            s[b + 1] = py - (jx - px)
                            s[b + 2] = (jr - 1) % 4
                        else:
                            width = nodes[j]["shapes"][jr][1]
                            s[b] = px + jy - py
                            s[b + 1] = py - (jx - px) - (width - 1)
                            s[b + 2] = (jr + 1) % 4
                    for j in descendant_dots[i]:
                        b = dot_offset + 2 * j
                        jx, jy = s[b], s[b + 1]
                        s[b] = px + jy - py
                        s[b + 1] = py - (jx - px)
                changed = True

        if not changed:
            return None
        result = tuple(s)
        return result if valid(result) else None

    return node_mask, transition, bars_by_color


def wall_distances(goals, walls):
    result = []
    for gx, gy in goals:
        dist = [1000] * 441
        if not (0 <= gx < 21 and 0 <= gy < 21):
            result.append(dist)
            continue
        q = deque([(gx, gy)])
        dist[gy * 21 + gx] = 0
        while q:
            x, y = q.popleft()
            d = dist[y * 21 + x] + 1
            for dx, dy in DIRS:
                xx, yy = x + dx, y + dy
                if not (0 <= xx < 21 and 0 <= yy < 21):
                    continue
                if (xx, yy) in walls:
                    continue
                k = yy * 21 + xx
                if d < dist[k]:
                    dist[k] = d
                    q.append((xx, yy))
        result.append(dist)
    return result


def joint_candidates(model, node_mask, bars_by_color, deadline):
    nodes = model["nodes"]
    parent = model["parent"]
    state = model["state"]
    dots = model["dots"]
    goals = model["goals"]

    rotate_colors = {
        c for c, kind, _, _ in model["actions"] if kind == "rotate"
    }
    resize_colors = {
        c for c, kind, _, _ in model["actions"] if kind == "resize"
    }
    if not rotate_colors or not goals or not dots:
        return None

    covered = set(goals) & set(dots)
    unsatisfied = [g for g in goals if g not in covered]
    if not unsatisfied:
        return None

    choices = []
    for j, dot in enumerate(dots):
        if dot in covered:
            continue
        owner = model["dot_owner"][j]
        if owner < 0 or nodes[owner]["kind"] != "bar":
            continue

        chain = []
        seen = set()
        i = owner
        while i != -1 and i not in seen:
            seen.add(i)
            if nodes[i]["kind"] != "bar":
                chain = []
                break
            chain.append(i)
            i = parent[i]
        chain.reverse()

        if not (2 <= len(chain) <= 4):
            continue
        if parent[chain[0]] != -1:
            continue
        colors = [nodes[i]["color"] for i in chain]
        if len(set(colors)) != len(colors):
            continue
        if any(len(bars_by_color.get(c, ())) != 1 for c in colors):
            continue
        if not any(c in rotate_colors for c in colors):
            continue
        if any(
            dots[k] in covered
            for k in model["descendant_dots"][chain[0]]
        ):
            continue

        ox, oy, r, length = state[4 * owner:4 * owner + 4]
        dx, dy = DIRS[r]
        offset = (dot[0] - ox) * dx + (dot[1] - oy) * dy
        if (ox + offset * dx, oy + offset * dy) != dot:
            continue
        marker_constant = offset - length
        for goal in unsatisfied:
            score = abs(dot[0] - goal[0]) + abs(dot[1] - goal[1])
            choices.append((score, j, goal, chain, marker_constant))

    if not choices:
        return None
    choices.sort(key=lambda item: item[0])
    _, marker, goal, chain, marker_constant = choices[0]
    chain_set = set(chain)
    rootx, rooty = state[4 * chain[0]:4 * chain[0] + 2]
    count = len(chain)

    obstacle_mask = 0
    for i in range(len(nodes)):
        if i in chain_set:
            continue
        a = 4 * i
        mask = node_mask(i, *state[a:a + 4])
        if mask < 0:
            return None
        obstacle_mask |= mask

    current_lengths = [state[4 * i + 3] for i in chain]
    current_dirs = [state[4 * i + 2] for i in chain]

    @lru_cache(maxsize=None)
    def axis_solutions(signs, target):
        if not signs:
            return ((),) if target == 0 else ()
        if len(signs) == 1:
            v = target * signs[0]
            return ((v,),) if 1 <= v <= 21 else ()
        out = []
        for prefix in itertools.product(range(1, 22), repeat=len(signs) - 1):
            remaining = target - sum(
                s * v for s, v in zip(signs[:-1], prefix)
            )
            last = remaining * signs[-1]
            if 1 <= last <= 21:
                out.append(prefix + (last,))
        return tuple(out)

    def orientation_cost(directions):
        cost = (current_dirs[0] - directions[0]) % 4
        for k in range(1, count):
            old = (current_dirs[k] - current_dirs[k - 1]) % 4
            new = (directions[k] - directions[k - 1]) % 4
            if old not in REL_INDEX or new not in REL_INDEX:
                return 1000
            cost += (REL_INDEX[new] - REL_INDEX[old]) % 3
        return cost

    candidates = []
    examined = 0
    for directions in itertools.product(range(4), repeat=count):
        if time.monotonic() >= deadline:
            break
        if any((directions[k] - directions[k - 1]) % 4 == 2
               for k in range(1, count)):
            continue

        possible = True
        for k, i in enumerate(chain):
            if nodes[i]["color"] not in rotate_colors:
                if k == 0:
                    if directions[k] != current_dirs[k]:
                        possible = False
                        break
                else:
                    old = (current_dirs[k] - current_dirs[k - 1]) % 4
                    new = (directions[k] - directions[k - 1]) % 4
                    if old != new:
                        possible = False
                        break
        if not possible:
            continue

        ldx, ldy = DIRS[directions[-1]]
        tx = goal[0] - rootx - marker_constant * ldx
        ty = goal[1] - rooty - marker_constant * ldy
        xi = [k for k, r in enumerate(directions) if DIRS[r][0]]
        yi = [k for k, r in enumerate(directions) if DIRS[r][1]]
        xs = axis_solutions(tuple(DIRS[directions[k]][0] for k in xi), tx)
        ys = axis_solutions(tuple(DIRS[directions[k]][1] for k in yi), ty)
        if not xs or not ys:
            continue

        group = []
        for xv in xs:
            for yv in ys:
                examined += 1
                if examined % 2048 == 0 and time.monotonic() >= deadline:
                    break
                lengths = [0] * count
                for k, v in zip(xi, xv):
                    lengths[k] = v
                for k, v in zip(yi, yv):
                    lengths[k] = v
                if any(
                    nodes[chain[k]]["color"] not in resize_colors
                    and lengths[k] != current_lengths[k]
                    for k in range(count)
                ):
                    continue

                px, py = rootx, rooty
                occupied = obstacle_mask
                okay = True
                for k, i in enumerate(chain):
                    mask = node_mask(i, px, py, directions[k], lengths[k])
                    if mask < 0 or mask & occupied:
                        okay = False
                        break
                    occupied |= mask
                    dx, dy = DIRS[directions[k]]
                    px += dx * lengths[k]
                    py += dy * lengths[k]
                if okay:
                    score = orientation_cost(directions) + sum(
                        abs(lengths[k] - current_lengths[k])
                        for k in range(count)
                    )
                    group.append((score, tuple(directions), tuple(lengths)))
            if time.monotonic() >= deadline:
                break
        group.sort()
        candidates.extend(group[:5])

    if not candidates:
        return None
    candidates.sort()
    return chain, candidates[:80]


def plan_joint(model, physics, history, deadline):
    node_mask, transition, bars_by_color = physics
    info = joint_candidates(
        model, node_mask, bars_by_color,
        min(deadline, time.monotonic() + 1.8)
    )
    if info is None:
        return None

    chain, candidates = info
    nodes = model["nodes"]
    start = model["state"]
    dot_offset = 4 * len(nodes)
    colors = {nodes[i]["color"] for i in chain}
    actions = [a for a in model["actions"] if a[0] in colors]
    if not actions:
        return None

    prepared = []
    for _, directions, lengths in candidates:
        rels = [0] * len(chain)
        for k in range(1, len(chain)):
            rels[k] = REL_INDEX[
                (directions[k] - directions[k - 1]) % 4
            ]
        prepared.append((directions[0], tuple(rels), lengths))

    color_index = {nodes[i]["color"]: k for k, i in enumerate(chain)}

    def scores(state):
        dirs = [state[4 * i + 2] for i in chain]
        lengths = [state[4 * i + 3] for i in chain]
        rels = [0] * len(chain)
        for k in range(1, len(chain)):
            rel = (dirs[k] - dirs[k - 1]) % 4
            if rel not in REL_INDEX:
                return [(1000, p) for p in prepared[:1]]
            rels[k] = REL_INDEX[rel]

        out = []
        for p in prepared:
            root_dir, target_rels, target_lengths = p
            cost = (dirs[0] - root_dir) % 4
            for k in range(len(chain)):
                cost += abs(lengths[k] - target_lengths[k])
                if k:
                    cost += (target_rels[k] - rels[k]) % 3
            out.append((cost, p))
        out.sort(key=lambda item: item[0])
        return out

    def solved(state):
        positions = {
            (state[dot_offset + 2 * j], state[dot_offset + 2 * j + 1])
            for j in range(len(model["dots"]))
        }
        return all(g in positions for g in model["goals"])

    start_sig = signature(model, start)
    forbidden = {tuple(click) for click in history.get(start_sig, [])}
    serial = itertools.count()
    h0 = scores(start)[0][0]
    heap = [(2.2 * h0, h0, next(serial), 0, start)]
    distance = {start: 0}
    previous = {}
    best_state = start
    best_quality = h0 + 2 * min(3, len(forbidden))
    found = None
    expanded = 0

    while heap:
        _, h, _, g, state = heapq.heappop(heap)
        if distance.get(state) != g:
            continue
        if solved(state):
            found = state
            break

        ranked = scores(state)
        h = ranked[0][0]
        if state != start and h <= best_quality + 2:
            sig = signature(model, state)
            quality = h + 2 * min(3, len(history.get(sig, [])))
            if quality < best_quality:
                best_state = state
                best_quality = quality

        expanded += 1
        if expanded % 32 == 0:
            if time.monotonic() >= deadline or expanded >= 35000:
                break

        for action in actions:
            if state == start and action[3] in forbidden:
                continue

            if action[1] == "rotate":
                nxt = transition(state, action)
                options = [] if nxt is None else [(nxt, 1)]
            else:
                k = color_index[action[0]]
                i = chain[k]
                desired = {p[2][k] for _, p in ranked[:8]}
                desired.add(1)
                current_length = state[4 * i + 3]
                desired.add(current_length + action[2])
                options = []
                nxt = state
                last_valid = None
                for steps in range(1, 22):
                    nxt = transition(nxt, action)
                    if nxt is None:
                        break
                    last_valid = (nxt, steps)
                    if nxt[4 * i + 3] in desired:
                        options.append((nxt, steps))
                if last_valid is not None and (
                    not options or options[-1][0] != last_valid[0]
                ):
                    options.append(last_valid)

            for nxt, cost in options:
                ng = g + cost
                if ng >= distance.get(nxt, 10**9):
                    continue
                distance[nxt] = ng
                previous[nxt] = (state, action, cost)
                nh = scores(nxt)[0][0]
                penalty = 0
                if ng <= 3:
                    nsig = signature(model, nxt)
                    penalty = 2 * min(3, len(history.get(nsig, [])))
                heapq.heappush(
                    heap,
                    (ng + 2.2 * nh + penalty,
                     nh, next(serial), ng, nxt)
                )

    end = found if found is not None else best_state
    if end == start:
        best = None
        for state, g in distance.items():
            if state == start:
                continue
            sig = signature(model, state)
            score = (
                scores(state)[0][0]
                + 2 * min(3, len(history.get(sig, [])))
                + 0.1 * g
            )
            candidate = (score, g, state)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            end = best[2]

    macros = []
    while end != start and end in previous:
        old, action, count = previous[end]
        macros.append((action, count))
        end = old
    macros.reverse()

    path = []
    state = start
    for action, count in macros:
        for _ in range(count):
            state = transition(state, action)
            if state is None:
                return path
            path.append((action, state))
    return path


def plan_general(model, physics, history, deadline):
    _, transition, bars_by_color = physics
    nodes = model["nodes"]
    parent = model["parent"]
    goals = model["goals"]
    actions = model["actions"]
    n = len(nodes)
    ndots = len(model["dots"])
    dot_offset = 4 * n

    if not goals or ndots < len(goals):
        return []

    rotate_colors = {
        c for c, kind, _, _ in actions if kind == "rotate"
    }
    resize_colors = {
        c for c, kind, _, _ in actions if kind == "resize"
    }

    dot_paths = []
    dot_rotators = []
    for owner in model["dot_owner"]:
        path = []
        rotators = []
        i = owner
        seen = set()
        while i != -1 and i not in seen:
            seen.add(i)
            if nodes[i]["kind"] == "bar":
                path.append(i)
                if nodes[i]["color"] in rotate_colors:
                    rotators.append(i)
            i = parent[i]
        dot_paths.append(path)
        dot_rotators.append(rotators)

    walls = set()
    for i, node in enumerate(nodes):
        if (node["kind"] == "fixed" and node["color"] == 15
                and parent[i] == -1):
            x, y, _, _ = node["initial"]
            for xx, yy in node["shapes"][0][0]:
                walls.add((x + xx, y + yy))
    distances = wall_distances(goals, walls)

    max_moved = 1
    for color, _, _, _ in actions:
        affected = set()
        for i in bars_by_color.get(color, ()):
            affected.update(model["descendant_dots"][i])
        max_moved = max(max_moved, len(affected))

    assignments = []
    if len(goals) <= 4 and ndots <= 6:
        assignments = list(
            itertools.permutations(range(ndots), len(goals))
        )

    def solved(state):
        positions = {
            (state[dot_offset + 2 * j], state[dot_offset + 2 * j + 1])
            for j in range(ndots)
        }
        return all(g in positions for g in goals)

    def dgoal(k, x, y):
        if not (0 <= x < 21 and 0 <= y < 21):
            return 1000
        d = distances[k][y * 21 + x]
        if d >= 1000:
            gx, gy = goals[k]
            return 60 + abs(x - gx) + abs(y - gy)
        return d

    def heuristic(state):
        costs = []
        for j in range(ndots):
            x = state[dot_offset + 2 * j]
            y = state[dot_offset + 2 * j + 1]
            candidates = [(x, y, 0)]
            for i in dot_rotators[j]:
                px, py = state[4 * i], state[4 * i + 1]
                xx, yy = x, y
                for turns in range(1, 4):
                    xx, yy = px + yy - py, py - (xx - px)
                    if (xx, yy) not in walls:
                        candidates.append((xx, yy, turns))

            can_x = False
            can_y = False
            if not dot_rotators[j]:
                for i in dot_paths[j]:
                    if nodes[i]["color"] in resize_colors:
                        dx, dy = DIRS[state[4 * i + 2]]
                        can_x |= bool(dx)
                        can_y |= bool(dy)

            row = []
            for k, (gx, gy) in enumerate(goals):
                if not dot_rotators[j]:
                    if (not can_x and x != gx) or (not can_y and y != gy):
                        row.append(250.0)
                        continue
                direct = dgoal(k, x, y)
                optimistic = min(
                    extra + dgoal(k, xx, yy)
                    for xx, yy, extra in candidates
                )
                row.append(0.65 * direct + 0.35 * optimistic)
            costs.append(row)

        if assignments:
            best = min(
                sum(costs[a[k]][k] for k in range(len(goals)))
                for a in assignments
            )
        else:
            best = sum(
                min(costs[j][k] for j in range(ndots))
                for k in range(len(goals))
            )
        return best / max_moved

    start = model["state"]
    start_sig = signature(model, start)
    forbidden = {tuple(click) for click in history.get(start_sig, [])}
    usable = [a for a in actions if a[0] in bars_by_color]
    if not usable:
        return []

    serial = itertools.count()
    h0 = heuristic(start)
    heap = [(2.3 * h0, h0, next(serial), 0, start)]
    distance = {start: 0}
    previous = {}
    best_state = start
    best_quality = h0 + 2 * min(3, len(forbidden))
    found = None
    expanded = 0

    while heap:
        _, h, _, g, state = heapq.heappop(heap)
        if distance.get(state) != g:
            continue
        if solved(state):
            found = state
            break
        if state != start and h <= best_quality + 2:
            sig = signature(model, state)
            quality = h + 2 * min(3, len(history.get(sig, [])))
            if quality < best_quality:
                best_state = state
                best_quality = quality

        expanded += 1
        if expanded % 128 == 0:
            if time.monotonic() >= deadline or expanded >= 100000:
                break

        for action in usable:
            if state == start and action[3] in forbidden:
                continue
            nxt = transition(state, action)
            if nxt is None:
                continue
            ng = g + 1
            if ng >= distance.get(nxt, 10**9):
                continue
            distance[nxt] = ng
            previous[nxt] = (state, action)
            nh = heuristic(nxt)
            penalty = 0
            if ng <= 3:
                nsig = signature(model, nxt)
                penalty = 2 * min(3, len(history.get(nsig, [])))
            heapq.heappush(
                heap,
                (ng + 2.3 * nh + penalty, nh, next(serial), ng, nxt)
            )

    end = found if found is not None else best_state
    if end == start:
        best = None
        for state, g in distance.items():
            if state == start:
                continue
            sig = signature(model, state)
            score = (
                heuristic(state)
                + 2 * min(3, len(history.get(sig, [])))
                + 0.1 * g
            )
            candidate = (score, g, state)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            end = best[2]

    path = []
    while end != start and end in previous:
        old, action = previous[end]
        path.append((action, end))
        end = old
    path.reverse()
    return path


def remember_action(memory, sig, row, col, expected):
    history = memory.setdefault("history", {})
    clicks = history.setdefault(sig, [])
    click = [int(row), int(col)]
    if click not in clicks:
        clicks.append(click)
    if len(history) > 250:
        for key in list(history)[:50]:
            del history[key]
    memory["last"] = [sig, int(row), int(col), expected]


def act(obs, available, memory):
    model = parse_frame(obs)
    current = signature(model, model["state"])
    history = memory.setdefault("history", {})

    last = memory.pop("last", None)
    if last is not None and current != last[3]:
        memory.pop("plan", None)

    recent = memory.setdefault("recent", [])
    recent.append(current)
    del recent[:-24]

    has_rotation = any(
        kind == "rotate" for _, kind, _, _ in model["actions"]
    )
    if (not has_rotation and not memory.get("reset_used")
            and len(recent) >= 16
            and len(set(recent[-16:])) <= 8
            and "RESET" in available):
        memory.clear()
        memory["reset_used"] = True
        return ("RESET", None, None)

    plan = memory.get("plan", [])
    if plan:
        row, col, expected = plan[0]
        if [row, col] in history.get(current, []):
            memory.pop("plan", None)
        else:
            plan.pop(0)
            memory["plan"] = plan
            remember_action(memory, current, row, col, expected)
            return ("ACTION6", int(row), int(col))

    physics = make_physics(model)
    deadline = time.monotonic() + 8.3
    path = None

    if has_rotation:
        path = plan_joint(model, physics, history, deadline)
    if path is None:
        path = plan_general(model, physics, history, deadline)

    if path:
        plan = [
            [action[3][0], action[3][1], signature(model, state)]
            for action, state in path
        ]
        row, col, expected = plan.pop(0)
        memory["plan"] = plan
        remember_action(memory, current, row, col, expected)
        return ("ACTION6", int(row), int(col))

    tried = {tuple(click) for click in history.get(current, [])}
    for _, _, _, (row, col) in model["actions"]:
        if (row, col) not in tried:
            remember_action(memory, current, row, col, "")
            return ("ACTION6", int(row), int(col))

    if "RESET" in available and not memory.get("reset_used"):
        memory.clear()
        memory["reset_used"] = True
        return ("RESET", None, None)

    return ("ACTION6", 0, 0)

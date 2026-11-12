import numpy as np
import heapq
import time
from collections import deque

# Actions: up, down, left, right.
_DELTAS = ((0, -6), (0, 6), (-6, 0), (6, 0))
_OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}
_ENEMY_ACTION = (2, 4, 1, 3)  # down, right, up, left


def _rings(im, color):
    result = []
    for y in range(9, 61):
        for x in range(2, 62):
            if im[y, x] != 5:
                continue
            if (im[y - 1, x] != color or im[y + 1, x] != color
                    or im[y, x - 1] != color or im[y, x + 1] != color):
                continue
            p = im[y - 2:y + 3, x - 2:x + 3]
            if p.shape == (5, 5) and np.count_nonzero(p == color) == 24:
                result.append((x, y))
    return result


def _components(im):
    labels = np.full((64, 64), -1, dtype=np.int16)
    number = 0
    for y in range(7, 63):
        for x in range(64):
            color = int(im[y, x])
            if color not in (8, 11, 15) or labels[y, x] >= 0:
                continue
            labels[y, x] = number
            q = [(x, y)]
            for xx, yy in q:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = xx + dx, yy + dy
                    if (0 <= nx < 64 and 7 <= ny < 63
                            and labels[ny, nx] < 0
                            and im[ny, nx] == color):
                        labels[ny, nx] = number
                        q.append((nx, ny))
            number += 1
    return labels


def _entities(im, centers):
    gates, buttons, pads = [], [], []
    for x, y in centers:
        if not (3 <= x < 61 and 9 <= y < 60):
            continue
        c = int(im[y, x])
        p = im[y - 2:y + 3, x - 2:x + 3]
        if c in (8, 11, 15):
            count = int(np.count_nonzero(p == c))
            if count >= 19:
                direction = None
                for a, (dx, dy) in enumerate(_DELTAS, 1):
                    if im[y + dy // 2, x + dx // 2] == c:
                        direction = a
                        break
                if direction is not None:
                    gates.append((x, y, c, direction))
            elif np.all(im[y - 1:y + 2, x - 1:x + 2] == c):
                buttons.append((x, y, c))

        if (np.all(im[y - 3, x - 3:x + 4] == 15)
                and np.all(im[y + 3, x - 3:x + 4] == 15)
                and np.all(im[y - 3:y + 4, x - 3] == 15)
                and np.all(im[y - 3:y + 4, x + 3] == 15)):
            pads.append((x, y))
    return gates, buttons, pads


def _find_goal(im, centers):
    required = set()
    for k in range(-3, 4):
        required.add((k, -3))
        required.add((k, 3))
        required.add((3, k))
    required.add((0, 0))
    rotations = []
    for r in range(4):
        shape = []
        for xx, yy in required:
            for _ in range(r):
                xx, yy = -yy, xx
            shape.append((xx, yy))
        rotations.append(shape)
    for x, y in centers:
        if not (4 <= x < 60 and 10 <= y < 60) or im[y, x] != 9:
            continue
        for shape in rotations:
            if all(im[y + dy, x + dx] == 9 for dx, dy in shape):
                return x, y
    return None


def _parse(im):
    players = _rings(im, 9)
    if not players:
        return None
    start_xy = players[0]
    ox, oy = start_xy[0] % 6, start_xy[1] % 6
    centers = [(x, y) for y in range(oy, 63, 6)
               for x in range(ox, 64, 6) if y >= 7 and 2 <= x <= 61]
    gates, buttons, pads = _entities(im, centers)
    goal_xy = _find_goal(im, centers)
    if goal_xy is None:
        return None
    enemy_xy = _rings(im, 14)

    floor = set()
    for x, y in centers:
        p = im[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        if p.shape == (5, 5) and im[y, x] != 0:
            if np.count_nonzero(p) >= 20:
                floor.add((x, y))
    floor.update((x, y) for x, y, _, _ in gates)
    floor.update((x, y) for x, y, _ in buttons)
    floor.update(pads)
    floor.update(enemy_xy)
    floor.add(start_xy)
    floor.add(goal_xy)

    xy = sorted(floor, key=lambda p: (p[1], p[0]))
    index = {p: i for i, p in enumerate(xy)}
    nb = []
    for x, y in xy:
        nb.append(tuple(index.get((x + dx, y + dy), -1)
                        for dx, dy in _DELTAS))

    labels = _components(im)
    circuit_labels = []
    for x, y, c in buttons:
        lab = int(labels[y, x])
        if lab not in circuit_labels:
            circuit_labels.append(lab)
    circuit_index = {lab: i for i, lab in enumerate(circuit_labels)}
    outputs = [[] for _ in circuit_labels]
    portal_outputs = [[] for _ in circuit_labels]

    gate_data = []
    for j, (x, y, c, a) in enumerate(gates):
        dx, dy = _DELTAS[a - 1]
        closed = index[(x, y)]
        opened = index.get((x + dx, y + dy), -1)
        gate_data.append((closed, opened, c == 11, x, y, c, a))
        lab = int(labels[y, x])
        ci = circuit_index.get(lab)
        if ci is not None:
            outputs[ci].append(j)

    button_nodes, button_colors, button_circuits = [], [], []
    at_button = [-1] * len(xy)
    for j, (x, y, c) in enumerate(buttons):
        node = index[(x, y)]
        button_nodes.append(node)
        button_colors.append(c)
        button_circuits.append(circuit_index[int(labels[y, x])])
        at_button[node] = j

    pad_groups = {}
    for x, y in pads:
        lab = int(labels[y - 3, x - 3])
        pad_groups.setdefault(lab, []).append(index[(x, y)])
    portal_pairs = []
    for lab, group in pad_groups.items():
        ci = circuit_index.get(lab)
        remaining = list(group)
        while len(remaining) >= 2:
            best = None
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    u, v = remaining[i], remaining[j]
                    x1, y1 = xy[u]
                    x2, y2 = xy[v]
                    score = abs(x1 - x2) + abs(y1 - y2)
                    if x1 != x2 and y1 != y2:
                        score += 100
                    if best is None or score < best[0]:
                        best = (score, i, j)
            _, i, j = best
            pair = (remaining[i], remaining[j])
            portal_pairs.append(pair)
            if ci is not None:
                portal_outputs[ci].append(pair)
            remaining.pop(j)
            remaining.pop(i)

    enemy_nodes = [index[p] for p in enemy_xy]
    tracks = []
    for node in enemy_nodes:
        seen = {node}
        q = [node]
        for u in q:
            for v in nb[u]:
                if v >= 0 and v not in seen:
                    seen.add(v)
                    q.append(v)
        tracks.append(seen)

    slots = 0
    for j in range(12):
        x = 4 * j + 1
        if x + 2 >= 64:
            break
        if int(im[1, x]) in (1, 2, 9) and np.all(im[1, x:x + 3] == im[1, x]):
            slots += 1
        else:
            break
    slots = max(2, slots)

    return {
        "xy": xy, "index": index, "nb": nb, "centers": centers,
        "start": index[start_xy], "goal": index[goal_xy],
        "gates": gate_data, "buttons": button_nodes,
        "button_colors": button_colors, "bc": button_circuits,
        "at_button": at_button, "outputs": outputs,
        "portal_outputs": portal_outputs, "pairs": portal_pairs,
        "enemy_start": enemy_nodes, "tracks": tracks,
        "slots": slots, "bad": set(), "distances": {},
        "blocked_cache": {},
    }


def _initial(board, programs, directions=None):
    enemies = []
    for i, node in enumerate(board["enemy_start"]):
        d = directions[i] if directions and i < len(directions) else 0
        enemies.extend((node, d))
    return (board["start"], 0, 0,
            tuple(board["start"] for _ in programs), tuple(enemies))


def _blocked(board, mask):
    cache = board["blocked_cache"]
    if mask not in cache:
        s = set()
        for j, gate in enumerate(board["gates"]):
            node = gate[1] if mask & (1 << j) else gate[0]
            if node >= 0:
                s.add(node)
        cache[mask] = frozenset(s)
    return cache[mask]


class _Simulation:
    def __init__(self, board, programs):
        self.b = board
        self.programs = tuple(tuple(p) for p in programs)
        self.cap = max((len(p) for p in programs), default=0)
        self.cache = {}

    def step(self, state, t, action):
        key = (state, min(t, self.cap), action)
        if key in self.cache:
            return self.cache[key]
        answer = self._step(state, t, action)
        if len(self.cache) < 90000:
            self.cache[key] = answer
        return answer

    def _step(self, state, t, action):
        b = self.b
        p, gm, wm, ghosts, enemies = state
        blocked = _blocked(b, gm)
        bad = b["bad"]
        nb = b["nb"]
        dest = nb[p][action - 1]
        if dest < 0 or dest in blocked or dest in bad:
            return None

        ng = len(ghosts)
        ne = len(enemies) // 2
        old = [p] + list(ghosts)
        dead = []
        edirs = []
        for i in range(ne):
            ep, direction = enemies[2 * i:2 * i + 2]
            dead.append(ep < 0)
            old.append(ep if ep >= 0 else -ep - 1)
            edirs.append(direction)
        new = list(old)
        new[0] = dest
        moved = [False] * len(old)
        moved[0] = True

        for i, program in enumerate(self.programs):
            if t >= len(program):
                continue
            v = nb[old[i + 1]][program[t] - 1]
            if v >= 0 and v not in blocked and v not in bad:
                new[i + 1] = v
                moved[i + 1] = True

        for i in range(ne):
            if dead[i]:
                continue
            ai = 1 + ng + i
            u = old[ai]
            direction = edirs[i]
            track = b["tracks"][i]
            for _ in range(4):
                choice = None
                for d in (direction, (direction - 1) % 4,
                          (direction + 1) % 4, (direction + 2) % 4):
                    v = nb[u][_ENEMY_ACTION[d] - 1]
                    if v >= 0 and v in track and v not in bad:
                        choice = (v, d)
                        break
                if choice is None:
                    v = u
                else:
                    v, direction = choice
                if v not in blocked:
                    new[ai] = v
                    moved[ai] = True
                    break
                direction = (direction + 2) % 4
            edirs[i] = direction

        at = b["at_button"]
        occupancy = [0] * len(b["buttons"])
        for i, u in enumerate(old):
            button = at[u]
            if button >= 0:
                occupancy[button] |= 1 << i

        masks = [gm, wm]
        closures = set()
        swaps = []

        def signal(button, value):
            ci = b["bc"][button]
            bit = 1 << ci
            if bool(masks[1] & bit) == value:
                return
            if value:
                masks[1] |= bit
            else:
                masks[1] &= ~bit
            for j in b["outputs"][ci]:
                gate = b["gates"][j]
                gbit = 1 << j
                was = bool(masks[0] & gbit)
                if gate[2]:
                    if not value:
                        continue
                    now = not was
                else:
                    now = value
                if now:
                    masks[0] |= gbit
                else:
                    masks[0] &= ~gbit
                    if was:
                        closures.add(gate[0])
            if value:
                swaps.extend(b["portal_outputs"][ci])

        for i, u in enumerate(old):
            if not moved[i]:
                continue
            button = at[u]
            if button >= 0:
                before = occupancy[button]
                occupancy[button] &= ~(1 << i)
                if before and not occupancy[button]:
                    signal(button, False)

        if new[0] in closures:
            return None
        for i in range(ne):
            ai = ng + 1 + i
            if new[ai] in closures:
                dead[i] = True
            if moved[ai] and not dead[i] and new[ai] == new[0]:
                return None
        closures.clear()

        # The engine delivers arrival signals in this order.
        order = [0] + list(range(ng + 1, len(new))) + list(range(1, ng + 1))
        for i in order:
            button = at[new[i]]
            if button >= 0:
                before = occupancy[button]
                occupancy[button] |= 1 << i
                if not before:
                    signal(button, True)

        if new[0] in closures:
            return None
        for i in range(ne):
            if new[ng + 1 + i] in closures:
                dead[i] = True

        for u, v in swaps:
            for i, pos in enumerate(new):
                if pos == u:
                    new[i] = v
                elif pos == v:
                    new[i] = u

        out_enemies = []
        for i in range(ne):
            pos = new[ng + 1 + i]
            out_enemies.extend((-pos - 1 if dead[i] else pos, edirs[i]))
        return (new[0], masks[0], masks[1],
                tuple(new[1:ng + 1]), tuple(out_enemies))

    def follow(self, state, t, path):
        for a in path:
            state = self.step(state, t, a)
            if state is None:
                return None
            t += 1
        return state


def _distances(board, target):
    cache = board["distances"]
    if target in cache:
        return cache[target]
    n = len(board["xy"])
    dist = [10000] * n
    dist[target] = 0
    extra = {}
    for u, v in board["pairs"]:
        extra.setdefault(u, []).append(v)
        extra.setdefault(v, []).append(u)
    q = deque([target])
    while q:
        u = q.popleft()
        d = dist[u]
        for v in extra.get(u, ()):
            if dist[v] > d:
                dist[v] = d
                q.appendleft(v)
        for v in board["nb"][u]:
            if v >= 0 and dist[v] > d + 1:
                dist[v] = d + 1
                q.append(v)
    cache[target] = dist
    return dist


def _search(sim, initial, initial_t, target, deadline,
            max_nodes=24000, max_depth=90, weight=1.15):
    b = sim.b
    dist = _distances(b, target)
    if dist[initial[0]] >= 10000:
        return None
    if initial[0] == target:
        return [], initial
    cap = sim.cap
    first_key = (min(initial_t, cap), initial)
    best = {first_key: 0}
    nodes = [(initial, -1, 0, 0)]
    heap = [(weight * dist[initial[0]], 0, 0)]
    expanded = 0
    while heap and expanded < max_nodes:
        if expanded % 64 == 0 and time.monotonic() >= deadline:
            return None
        _, g, ni = heapq.heappop(heap)
        state, parent, action, _ = nodes[ni]
        k = (min(initial_t + g, cap), state)
        if best.get(k) != g:
            continue
        if state[0] == target:
            path = []
            while nodes[ni][1] >= 0:
                path.append(nodes[ni][2])
                ni = nodes[ni][1]
            path.reverse()
            return path, state
        if g >= max_depth:
            continue
        expanded += 1
        candidates = []
        for a in (1, 2, 3, 4):
            v = b["nb"][state[0]][a - 1]
            if v >= 0:
                candidates.append((dist[v], a))
        candidates.sort()
        for _, a in candidates:
            nxt = sim.step(state, initial_t + g, a)
            if nxt is None:
                continue
            ng = g + 1
            nk = (min(initial_t + ng, cap), nxt)
            if ng >= best.get(nk, 10**9):
                continue
            best[nk] = ng
            j = len(nodes)
            nodes.append((nxt, ni, a, ng))
            priority = ng + weight * dist[nxt[0]]
            heapq.heappush(heap, (priority, ng, j))
    return None


class _Planner:
    def __init__(self, board, deadline):
        self.b = board
        self.deadline = deadline
        self.contexts = {}
        self.fallback = None
        self.visited = set()

    def context(self, programs):
        key = tuple(tuple(p) for p in programs)
        if key not in self.contexts:
            self.contexts[key] = _Simulation(self.b, key)
        return self.contexts[key]

    def delayed(self, sim, state, t, path, delay, target):
        if delay == 0:
            end = sim.follow(state, t, path)
            return (path, end) if end is not None and end[0] == target else None
        points = []
        for i in (0, 1, len(path) // 2, max(0, len(path) - 1)):
            if 0 <= i <= len(path) and i not in points:
                points.append(i)
        for i in points:
            prefix_state = sim.follow(state, t, path[:i])
            if prefix_state is None:
                continue
            for a in (1, 2, 3, 4):
                loop = [a, _OPPOSITE[a]] * (delay // 2)
                trial = path[:i] + loop + path[i:]
                end = sim.follow(state, t, trial)
                if end is not None and end[0] == target:
                    return trial, end
        return None

    def solve(self, programs, state, history, used=(), root=True):
        if time.monotonic() >= self.deadline:
            return None
        signature = (tuple(tuple(p) for p in programs), state, tuple(history))
        if signature in self.visited:
            return None
        self.visited.add(signature)

        sim = self.context(programs)
        t = len(history)
        remaining_slots = self.b["slots"] - 1 - len(programs)
        allowance = 1.8 if root else 0.48
        if root and remaining_slots <= 0:
            allowance = 5.8
        result = _search(
            sim, state, t, self.b["goal"],
            min(self.deadline, time.monotonic() + allowance),
            max_nodes=75000 if root else 26000,
            max_depth=max(15, 118 - t),
        )
        if result is not None:
            return result[0]
        if remaining_slots <= 0 or time.monotonic() >= self.deadline:
            return None

        candidates = []
        for j, node in enumerate(self.b["buttons"]):
            if node in used:
                continue
            ci = self.b["bc"][j]
            if not self.b["outputs"][ci] and not self.b["portal_outputs"][ci]:
                continue
            color = self.b["button_colors"][j]
            rank = 0 if color == 8 else (1 if color == 15 else 2)
            candidates.append((rank, _distances(self.b, node)[state[0]], j, node))
        candidates.sort()

        routes = []
        for rank, _, j, node in candidates:
            if time.monotonic() >= self.deadline:
                break
            result = _search(
                sim, state, t, node,
                min(self.deadline, time.monotonic() + 0.32),
                max_nodes=14000, max_depth=max(10, 68 - t), weight=1.05,
            )
            if result is None:
                continue
            path, end = result
            if not path and not history:
                continue
            routes.append((rank, j, node, path, end))
            if root and self.fallback is None:
                self.fallback = path + [5]

        # First try short recordings of every useful switch.
        trials = [(r, 0) for r in routes]
        # White circuits normally need the ghost to arrive at a precise time.
        for delay in range(2, 27, 2):
            for r in routes:
                ci = self.b["bc"][r[1]]
                if self.b["portal_outputs"][ci]:
                    trials.append((r, delay))
        # Delayed ordinary recordings can also control enemy timing.
        for delay in (2, 4, 6, 8):
            for r in routes:
                if r[0] == 0:
                    trials.append((r, delay))

        for route, delay in trials:
            if time.monotonic() >= self.deadline:
                break
            _, j, node, base_path, base_end = route
            if delay:
                variant = self.delayed(sim, state, t, base_path, delay, node)
                if variant is None:
                    continue
                path, end = variant
            else:
                path, end = base_path, base_end
            program = tuple(history) + tuple(path)
            if not program or len(program) > 85:
                continue
            new_programs = tuple(programs) + (program,)
            directions = list(end[4][1::2])
            new_state = _initial(self.b, new_programs, directions)
            subplan = self.solve(new_programs, new_state, [],
                                 tuple(used) + (node,), False)
            if subplan is not None:
                return path + [5] + subplan
        return None


def _observe_state(im, board, predicted):
    """Correct the visible parts of a simulated state."""
    p, gm, wm, ghosts, enemies = predicted
    index = board["index"]
    players = _rings(im, 9)
    if players and players[0] in index:
        p = index[players[0]]

    visible_gates, _, _ = _entities(im, board["centers"])
    gm2 = gm
    for j, gate in enumerate(board["gates"]):
        closed, opened, toggle, x, y, color, direction = gate
        dx, dy = _DELTAS[direction - 1]
        on = (x + dx, y + dy, color, direction)
        off = (x, y, color, direction)
        if on in visible_gates:
            gm2 |= 1 << j
        elif off in visible_gates:
            gm2 &= ~(1 << j)

    positions = [index[xy] for xy in _rings(im, 14) if xy in index]
    enemies2 = list(enemies)
    unused = set(range(len(positions)))
    for i in range(len(enemies) // 2):
        ep = enemies[2 * i]
        oldpos = ep if ep >= 0 else -ep - 1
        if unused:
            x, y = board["xy"][oldpos]
            k = min(unused, key=lambda z:
                    abs(board["xy"][positions[z]][0] - x)
                    + abs(board["xy"][positions[z]][1] - y))
            newpos = positions[k]
            unused.remove(k)
            enemies2[2 * i] = newpos
        else:
            enemies2[2 * i] = -oldpos - 1

    # Match visible ghosts conservatively: overlap often hides some of them.
    ghost_visible = [index[xy] for xy in _rings(im, 2) if xy in index]
    ghost_list = list(ghosts)
    unmatched = list(range(len(ghost_list)))
    for pos in ghost_visible:
        if not unmatched:
            break
        x, y = board["xy"][pos]
        k = min(unmatched, key=lambda z:
                abs(board["xy"][ghost_list[z]][0] - x)
                + abs(board["xy"][ghost_list[z]][1] - y))
        unmatched.remove(k)
        ghost_list[k] = pos

    if gm2 != gm:
        # Usually each circuit has one pressure plate. Reconstruct its signal
        # when an observed gate state reveals a prediction discrepancy.
        actors = [p] + ghost_list
        actors += [v if v >= 0 else -v - 1 for v in enemies2[::2]]
        occupied = set(actors)
        wm = 0
        for j, node in enumerate(board["buttons"]):
            if node in occupied:
                wm |= 1 << board["bc"][j]
    return (p, gm2, wm, tuple(ghost_list), tuple(enemies2))


def _choose_safe(board, sim, state, t, memory):
    visits = memory.setdefault("visits", {})
    dist = _distances(board, board["goal"])
    choices = []
    for a in (1, 2, 3, 4):
        nxt = sim.step(state, t, a)
        if nxt is None:
            continue
        k = (nxt[0], nxt[1], nxt[4])
        penalty = visits.get(k, 0)
        score = min(dist[nxt[0]], 200) + 8 * penalty
        choices.append((score, a, k))
    if choices:
        choices.sort()
        _, a, k = choices[0]
        visits[k] = visits.get(k, 0) + 1
        return a
    return 5 if t else 1


def act(obs, available, memory):
    im = np.asarray(obs)
    started = time.monotonic()

    # A reset also makes arbitrary supplied mid-level states fully observable:
    # it removes unknown recordings and restores the circuit initial state.
    if "board" not in memory:
        if not memory.get("requested_reset"):
            memory["requested_reset"] = True
            return ("RESET", None, None)
        board = _parse(im)
        if board is None:
            memory["parse_attempts"] = memory.get("parse_attempts", 0) + 1
            if memory["parse_attempts"] <= 2:
                return ("RESET", None, None)
            return ("ACTION1", None, None)
        memory["board"] = board
        memory["programs"] = ()
        memory["history"] = []
        memory["used"] = []
        memory["state"] = _initial(board, ())
        memory["plan"] = []
        memory["steps"] = 0

    board = memory["board"]
    programs = tuple(memory.get("programs", ()))
    predicted = memory["state"]
    observed = _observe_state(im, board, predicted)
    previous = memory.get("previous")
    last = memory.get("last")

    # Detect a tile mistakenly inferred from decorative wiring.
    if previous is not None and last in (1, 2, 3, 4):
        attempted = board["nb"][previous[0]][last - 1]
        if (observed[0] == previous[0] and predicted[0] != previous[0]
                and predicted[0] == attempted
                and observed[4][::2] == previous[4][::2]):
            board["bad"].add(attempted)
            memory["history"] = memory["history"][:memory.get("previous_t", 0)]
            observed = previous
            memory["plan"] = []

    if observed != predicted:
        memory["plan"] = []

        # Learn invisible enemy-track restrictions from unexpected turns.
        if previous is not None and last in (1, 2, 3, 4):
            olde, prede, reale = previous[4], predicted[4], observed[4]
            corrected = list(reale)
            for i in range(len(reale) // 2):
                oldpos = olde[2 * i]
                pp = prede[2 * i]
                rp = reale[2 * i]
                if oldpos < 0 or rp < 0:
                    continue
                if rp in board["nb"][oldpos]:
                    action = board["nb"][oldpos].index(rp) + 1
                    corrected[2 * i + 1] = _ENEMY_ACTION.index(action)
                    if pp >= 0 and pp != rp and pp != oldpos:
                        if pp not in _blocked(board, previous[1]):
                            board["tracks"][i].discard(pp)
            observed = (observed[0], observed[1], observed[2],
                        observed[3], tuple(corrected))

    memory["state"] = observed
    history = list(memory.get("history", []))
    plan = list(memory.get("plan", []))

    if not plan:
        planner = _Planner(board, started + 8.0)
        plan = planner.solve(programs, observed, history,
                             tuple(memory.get("used", [])), True)
        if plan is None:
            plan = planner.fallback
        if not plan:
            sim = _Simulation(board, programs)
            a = _choose_safe(board, sim, observed, len(history), memory)
            plan = [a]

    action = int(plan.pop(0))
    sim = _Simulation(board, programs)
    memory["previous"] = observed
    memory["previous_t"] = len(history)
    memory["last"] = action
    memory["steps"] = memory.get("steps", 0) + 1

    if action == 5:
        if history:
            if len(programs) + 1 >= board["slots"]:
                programs = ()
                memory["used"] = []
            else:
                programs = programs + (tuple(history),)
                memory.setdefault("used", []).append(observed[0])
            directions = list(observed[4][1::2])
            memory["state"] = _initial(board, programs, directions)
            memory["programs"] = programs
            memory["history"] = []
        else:
            # Rewinding an empty recording is a no-op.
            memory["state"] = observed
    else:
        nxt = sim.step(observed, len(history), action)
        if nxt is None:
            action = _choose_safe(board, sim, observed, len(history), memory)
            memory["last"] = action
            plan = []
            if action in (1, 2, 3, 4):
                nxt = sim.step(observed, len(history), action)
            else:
                nxt = None
        if action in (1, 2, 3, 4) and nxt is not None:
            history.append(action)
            memory["history"] = history
            memory["state"] = nxt
        elif action == 5 and history:
            if len(programs) + 1 >= board["slots"]:
                programs = ()
                memory["used"] = []
            else:
                programs = programs + (tuple(history),)
                memory.setdefault("used", []).append(observed[0])
            memory["programs"] = programs
            memory["history"] = []
            memory["state"] = _initial(board, programs, list(observed[4][1::2]))

    memory["plan"] = plan
    return ("ACTION" + str(action), None, None)

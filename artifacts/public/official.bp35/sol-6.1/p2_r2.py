import math
import heapq
import time
from collections import deque
import numpy as np

EMPTY, WALL, BREAK, GREEN, SOLID, GHOST, GROW, SPIKE, GOAL, LAVA = range(10)


def _decode(obs):
    ys, xs = np.where(obs[:63] == 9)
    if not len(ys):
        return None

    top = int(ys.min())
    px = int(round((float(np.median(xs[ys == top])) - 3) / 6))
    player_top = top - 1
    cells = {}

    for ry in range(math.ceil(-player_top / 6),
                    math.floor((56 - player_top) / 6) + 1):
        y = player_top + 6 * ry
        for x in range(11):
            left = 6 * x
            patch = obs[y + 2:y + 5, left + 2:min(left + 5, 64)]
            center = int(obs[y + 3, min(left + 3, 63)])

            if np.any(patch == 9):
                kind = EMPTY
            elif np.any(patch == 7):
                kind = GOAL
            elif center == 14:
                kind = BREAK
            elif center == 8:
                kind = GREEN
            elif center == 12:
                kind = SOLID if np.count_nonzero(patch == 12) >= 7 else GHOST
            elif center == 15:
                c = min(left + 3, 63)
                if obs[y + 1, c] == 0 or obs[y + 5, c] == 0:
                    kind = SPIKE
                else:
                    corners = [
                        int(obs[y + dy, left + dx])
                        for dy in (1, 5) for dx in (1, 5)
                        if left + dx < 64
                    ]
                    kind = GROW if 3 in corners else LAVA
            elif center == 10:
                kind = EMPTY
            elif np.any(patch == 15):
                kind = LAVA
            else:
                kind = WALL

            cells[(x, ry)] = kind

    return px, player_top, cells


def _align(cells, board, candidates, preferred, px):
    best = None
    for py in candidates:
        matches = errors = overlap = 0
        for (x, ry), value in cells.items():
            if x == px and ry == 0:
                continue
            old = board.get((x, py + ry))
            if old is None:
                continue
            weight = 1 if value in (EMPTY, WALL) and old in (EMPTY, WALL) else 4
            overlap += weight
            if old == value:
                matches += weight
            else:
                errors += weight

        if overlap < 22 or errors > 0.13 * overlap:
            continue
        score = matches - 7 * errors - 0.08 * abs(py - preferred)
        if best is None or score > best[0]:
            best = score, py

    return None if best is None else best[1]


def _key(px, py, gravity, edits):
    return px, py, gravity, tuple(sorted(edits.items()))


def _model(board, px, py, gravity, player_top, available):
    low = min(y for x, y in board)
    high = max(y for x, y in board)
    width = 11
    base = [-1] * ((high - low + 1) * width)
    special = []
    goals = []
    greens = []
    grows = []
    rows = set()

    for (x, y), kind in board.items():
        if 0 <= x < width:
            i = (y - low) * width + x
            base[i] = kind
            if kind in (BREAK, GREEN, SOLID, GHOST, GROW):
                special.append((x, y, i))
            if kind == GOAL:
                goals.append((x, y))
            elif kind == GREEN:
                greens.append((x, y))
            elif kind == GROW:
                grows.append((x, y))
            rows.add(y)

    root = (px, py, gravity, 0, 0, 0, py * 6 - player_top)

    def index(x, y):
        if x < 0 or x >= width or y < low or y > high:
            return -1
        return (y - low) * width + x

    def tile(state, x, y):
        if x < 0 or x >= width:
            return WALL
        i = index(x, y)
        if i < 0:
            return -1
        kind = base[i]
        bit = 1 << i
        if state[5] & bit:
            return GROW
        if kind in (BREAK, GREEN, GROW) and state[3] & bit:
            return EMPTY
        if kind in (SOLID, GHOST) and state[4] & bit:
            return GHOST if kind == SOLID else SOLID
        return kind

    def scan(state, x, y):
        g = state[2]
        last_y = y
        for _ in range(200):
            ny = last_y + g
            kind = tile(state, x, ny)
            if kind == -1:
                return last_y, 2
            if kind in (EMPTY, GHOST):
                last_y = ny
            elif kind == SPIKE:
                return last_y, 3
            elif kind == GOAL:
                return ny, 1
            else:
                return last_y, 0
        return last_y, 3

    def transition(state, action):
        x, y, g, removed, toggled, grown, camera = state

        if action[0] == "m":
            nx = x + action[1]
            kind = tile(state, nx, y)
            if kind == GOAL:
                return (nx, y, g, removed, toggled, grown, camera), 1
            if kind == -1:
                return (nx, y, g, removed, toggled, grown, camera), 2
            if kind not in (EMPTY, GHOST):
                return None, 3

            ny, result = scan(state, nx, y)
            if result == 3:
                return None, 3
            nc = camera
            if ny != y or result == 2:
                nc = ny * 6 - (36 if g == -1 else 26)
            return (nx, ny, g, removed, toggled, grown, nc), result

        tx, ty = action[1], action[2]
        i = index(tx, ty)
        if i < 0:
            return None, 3
        bit = 1 << i
        kind = tile(state, tx, ty)
        forced = False
        blocked_flip = False

        if kind == BREAK:
            removed |= bit
        elif kind in (SOLID, GHOST):
            toggled ^= bit
        elif kind == GREEN:
            g = -g
            forced = True
            support = tile(state, x, y + g)
            blocked_flip = support in (WALL, BREAK, GREEN, SOLID, GROW)
            removed |= bit
        elif kind == GROW:
            removed |= bit
            grown &= ~bit
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                xx, yy = tx + dx, ty + dy
                ii = index(xx, yy)
                if ii >= 0 and (xx, yy) != (x, y):
                    if tile(state, xx, yy) == EMPTY:
                        grown |= 1 << ii
        else:
            return None, 3

        new = (x, y, g, removed, toggled, grown, camera)

        if forced and blocked_flip:
            return (x, y, g, removed, toggled, grown,
                    y * 6 - (36 if g == -1 else 26)), 0

        if not forced and (tx != x or ty != y + g):
            return new, 0

        # Click-triggered falling starts inside the support cell.
        ny, result = scan(new, x, y + g)
        if result == 3:
            return None, 3
        nc = ny * 6 - (36 if g == -1 else 26)
        return (x, ny, g, removed, toggled, grown, nc), result

    strategic = bool(goals and greens and grows)

    def actions(state):
        x, y, g, removed, toggled, grown, camera = state
        if "ACTION3" in available:
            yield ("m", -1)
        if "ACTION4" in available:
            yield ("m", 1)
        if "ACTION6" not in available:
            return

        candidates = special
        if grown:
            candidates = list(special)
            bits = grown
            while bits:
                bit = bits & -bits
                i = bit.bit_length() - 1
                candidates.append((i % width, low + i // width, i))
                bits ^= bit

        emitted = set()
        for tx, ty, i in candidates:
            if (tx, ty) in emitted:
                continue
            emitted.add((tx, ty))
            if not 0 <= ty * 6 - camera + 3 < 63:
                continue

            kind = tile(state, tx, ty)
            side = ty == y and abs(tx - x) == 1
            support = tx == x and ty == y + g
            useful = kind == GREEN

            if kind in (BREAK, SOLID):
                useful = side or support
            elif kind == GHOST:
                useful = abs(tx - x) <= 1 and g * (ty - y) >= 0
            elif kind == GROW:
                if strategic:
                    useful = side or support
                else:
                    useful = -1 <= g * (ty - y) <= 6
                    if goals:
                        useful |= any(
                            abs(tx - gx) <= 3 and abs(ty - gy) <= 3
                            for gx, gy in goals
                        )

            if useful:
                yield ("c", tx, ty)

    def changes(state):
        result = {}
        bits = state[3] | state[4] | state[5]
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            x, y = i % width, low + i // width
            result[(x, y)] = tile(state, x, y)
            bits ^= bit
        return result

    def growth_macros(state):
        if not strategic or "ACTION6" not in available:
            return

        x, y, g = state[:3]
        camera = state[6]
        vlo = max(low, math.ceil((camera - 3) / 6))
        vhi = min(high, math.floor((camera + 59) / 6))

        parents = {}
        queue = deque()
        for yy in range(vlo, vhi + 1):
            for xx in range(1, 10):
                if (xx, yy) != (x, y) and tile(state, xx, yy) == GROW:
                    parents[(xx, yy)] = None
                    queue.append((xx, yy))

        if not queue:
            return

        distances = {p: 0 for p in parents}
        while queue:
            p = queue.popleft()
            if distances[p] >= 7:
                continue
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                q = p[0] + dx, p[1] + dy
                if q in parents or q == (x, y):
                    continue
                if not (1 <= q[0] <= 9 and vlo <= q[1] <= vhi):
                    continue
                if tile(state, *q) != EMPTY:
                    continue
                parents[q] = p
                distances[q] = distances[p] + 1
                queue.append(q)

        targets = set()

        # Landing supports and overhead stopping points near the actor.
        for dx in range(-2, 3):
            for k in range(1, 6):
                targets.add((x + dx, y + g * k))
        for dx in (-1, 0, 1):
            for k in range(2, 5):
                targets.add((x + dx, y - g * k))

        # Prepare the rows from which a horizontal goal approach is possible.
        for gx, gy in goals:
            for xx in range(1, 10):
                targets.add((xx, gy - 1))
                targets.add((xx, gy + 1))

        # Supports from which a remote gravity switch becomes visible.
        for gx, gy in greens:
            yy = gy - g * (6 if g == -1 else 4) + g
            for xx in range(1, 10):
                targets.add((xx, yy))

        ordered = [
            p for p in targets
            if p in parents and parents[p] is not None
            and tile(state, *p) == EMPTY
        ]
        ordered.sort(key=lambda p: (
            distances[p],
            min((abs(p[1] - gy) for gx, gy in goals), default=0),
            abs(p[0] - x)
        ))

        seen = set()
        for target in ordered[:48]:
            path = []
            p = target
            while p is not None:
                path.append(p)
                p = parents[p]
            path.reverse()
            path = path[:-1]

            current = state
            sequence = []
            status = 0
            valid = True
            for tx, ty in path:
                if tile(current, tx, ty) != GROW:
                    valid = False
                    break
                if not 0 <= ty * 6 - current[6] + 3 < 63:
                    valid = False
                    break
                action = ("c", tx, ty)
                nxt, status = transition(current, action)
                if nxt is None or status == 3:
                    valid = False
                    break
                sequence.append(action)
                current = nxt
                if status in (1, 2):
                    break

            if valid and sequence and current not in seen:
                seen.add(current)
                yield current, status, tuple(sequence)

        # Consecutive support clicks provide a compact mining/ascending move.
        current = state
        sequence = []
        for _ in range(18):
            xx, yy, gg = current[:3]
            if tile(current, xx, yy + gg) != GROW:
                break
            action = ("c", xx, yy + gg)
            nxt, status = transition(current, action)
            if nxt is None or status == 3:
                break
            sequence.append(action)
            current = nxt
            if len(sequence) >= 2 and current not in seen:
                seen.add(current)
                yield current, status, tuple(sequence)
            if status in (1, 2):
                break

    relaxed = None
    if strategic:
        passable = {
            p for p, kind in board.items()
            if kind not in (WALL, SPIKE, LAVA) and 0 <= p[0] < 11
        }

        def reverse_dist(seeds, g):
            dist = {}
            heap = []
            counter = 0
            for p, value in seeds.items():
                if p in passable and value < dist.get(p, float("inf")):
                    dist[p] = value
                    heapq.heappush(heap, (value, counter, p))
                    counter += 1

            while heap:
                value, _, p = heapq.heappop(heap)
                if value != dist.get(p):
                    continue
                for q, cost in (
                    ((p[0] - 1, p[1]), 1.0),
                    ((p[0] + 1, p[1]), 1.0),
                    ((p[0], p[1] - g), 0.18),
                ):
                    if q not in passable:
                        continue
                    nv = value + cost
                    if nv < dist.get(q, float("inf")):
                        dist[q] = nv
                        heapq.heappush(heap, (nv, counter, q))
                        counter += 1
            return dist

        direct = {
            g: reverse_dist({p: 0.0 for p in goals}, g)
            for g in (-1, 1)
        }
        flips = {}
        for g in (-1, 1):
            seeds = {}
            offset = 36 if g == -1 else 26
            for p in passable:
                for gx, gy in greens:
                    sy = (gy - p[1]) * 6 + offset + 3
                    if 0 <= sy < 63 and p in direct[-g]:
                        seeds[p] = min(
                            seeds.get(p, float("inf")),
                            1.0 + direct[-g][p]
                        )
            flips[g] = reverse_dist(seeds, g)
        relaxed = direct, flips

    return {
        "root": root,
        "tile": tile,
        "transition": transition,
        "actions": actions,
        "changes": changes,
        "macros": growth_macros,
        "goals": goals,
        "greens": greens,
        "rows": rows,
        "strategic": strategic,
        "relaxed": relaxed,
    }


def _search(model, visits, forbidden, deadline):
    root = model["root"]
    tile = model["tile"]
    transition = model["transition"]
    actions = model["actions"]
    goals = model["goals"]
    greens = model["greens"]
    known_rows = model["rows"]
    strategic = model["strategic"]
    rx, ry, rg = root[:3]

    def old_h(state):
        if not goals:
            return 0.0
        x, y, g = state[:3]
        return min(
            0.7 * abs(x - gx) + abs(y - gy) / 12.0
            + (3.5 if (gy - y) * g < 0 else 0.0)
            for gx, gy in goals
        )

    def heuristic(state):
        if not strategic:
            return old_h(state)
        x, y, g = state[:3]
        direct, flips = model["relaxed"]
        value = direct[g].get((x, y), float("inf"))
        if any(tile(state, gx, gy) == GREEN for gx, gy in greens):
            value = min(value, flips[g].get((x, y), float("inf")))
        if math.isinf(value):
            return 65.0 + old_h(state)
        return value

    def has_exit(state):
        x, y, g = state[:3]
        for dx in (-1, 1):
            nxt, status = transition(state, ("m", dx))
            if nxt is not None and status != 3:
                return True
            tx = x + dx
            if tile(state, tx, y) in (BREAK, SOLID, GROW):
                opened, status = transition(state, ("c", tx, y))
                if opened is not None and status == 0:
                    after, status = transition(opened, ("m", dx))
                    if after is not None and status != 3:
                        return True
        for action in actions(state):
            if action[0] == "c" and tile(state, action[1], action[2]) == GREEN:
                nxt, status = transition(state, action)
                if nxt is not None and status != 3:
                    return True
        return False

    parents = {}

    def route(state, tail=()):
        chunks = [tail] if tail else []
        while state != root and state in parents:
            parent, sequence = parents[state]
            chunks.append(sequence)
            state = parent
        result = []
        for chunk in reversed(chunks):
            result.extend(chunk)
        return result

    root_h = heuristic(root)
    weight = 2.8 if strategic else 1.0
    heap = [(weight * root_h, 0, 0, root)]
    costs = {root: 0}
    serial = expansions = 0
    limit = 100000 if goals else 60000
    depth_limit = 95 if strategic else (75 if goals else 30)
    safe = frontier = fallback = None
    exit_cache = {}

    while heap and expansions < limit:
        if expansions % 64 == 0 and time.perf_counter() >= deadline:
            break
        _, depth, _, state = heapq.heappop(heap)
        if costs.get(state) != depth:
            continue
        expansions += 1
        if depth >= depth_limit:
            continue

        successors = []
        for action in actions(state):
            if state == root and action in forbidden:
                continue
            nxt, status = transition(state, action)
            if nxt is not None:
                successors.append((nxt, status, (action,)))

        if strategic:
            for nxt, status, sequence in model["macros"](state):
                if state == root and sequence[0] in forbidden:
                    continue
                successors.append((nxt, status, sequence))

        for nxt, status, sequence in successors:
            nd = depth + len(sequence)
            if nd > depth_limit:
                continue
            if status == 1:
                return route(state, sequence), "win"

            x, y, g = nxt[:3]
            repeated = visits.get((x, y, g), 0)
            if nxt[:3] == root[:3]:
                repeated = 0

            if status == 2:
                score = (
                    2.0 / (nd + 0.5)
                    + 0.035 * min(abs(y - ry), 18)
                    - 0.025 * abs(x - rx)
                    - 0.12 * repeated
                )
                if goals:
                    score += 0.06 * (root_h - heuristic(nxt))
                if frontier is None or score > frontier[0]:
                    frontier = score, route(state, sequence)
                continue

            if costs.get(nxt, 10**9) <= nd:
                continue
            costs[nxt] = nd
            parents[nxt] = state, sequence

            camera = nxt[6]
            vlo = math.ceil(camera / 6)
            vhi = math.floor((camera + 56) / 6)
            gain = sum(row not in known_rows for row in range(vlo, vhi + 1))
            if gain:
                score = (
                    2.5 * gain / (nd + 1)
                    + 0.04 * min(abs(y - ry), 15)
                    - 0.025 * nd - 0.12 * repeated
                )
                if goals:
                    score += 0.04 * (root_h - heuristic(nxt))
                if safe is None or score > safe[0]:
                    live = exit_cache.get(nxt)
                    if live is None:
                        live = has_exit(nxt)
                        exit_cache[nxt] = live
                    if live:
                        safe = score, route(nxt)

            if goals:
                improvement = root_h - heuristic(nxt)
                score = improvement / (nd + 0.5) - 0.035 * nd - 0.12 * repeated
                if improvement > 0.2:
                    if fallback is None or score > fallback[0]:
                        fallback = score, route(nxt)

            serial += 1
            heapq.heappush(
                heap, (nd + weight * heuristic(nxt), nd, serial, nxt)
            )

    if strategic and fallback is not None:
        return fallback[1], "progress"
    if safe is not None:
        return safe[1], "explore"
    if frontier is not None:
        return frontier[1], "explore"
    if fallback is not None:
        return fallback[1], "progress"
    return [], "none"


def _update_edits(memory, changes):
    base, edits = memory["base"], memory["edits"]
    for key, value in changes.items():
        if base.get(key) == value:
            edits.pop(key, None)
        else:
            edits[key] = value


def act(obs, available, memory):
    deadline = time.perf_counter() + 8.3
    available = set(available)
    decoded = _decode(obs)

    if decoded is None:
        if "RESET" in available:
            memory.clear()
            return "RESET", None, None
        name = next((a for a in ("ACTION7", "ACTION3", "ACTION4")
                     if a in available), next(iter(available), "ACTION3"))
        return name, None, None

    px, player_top, cells = decoded
    base = memory.setdefault("base", {})
    edits = memory.setdefault("edits", {})
    history = memory.setdefault("history", [])
    banned = memory.setdefault("banned", {})
    counts = memory.setdefault("counts", {})
    visits = memory.setdefault("visits", {})
    prediction = memory.pop("prediction", None)
    plan = memory.pop("plan", [])

    board = dict(base)
    board.update(edits)
    previous = memory.get("last_picture")

    ineffective = (
        prediction is not None
        and prediction.get("action", (None,))[0] == "c"
        and previous is not None
        and np.array_equal(previous, obs[:63])
    )

    if ineffective and history:
        entry = history[-1]
        banned.setdefault(entry["key"], set()).add(entry["action"])
        edits.clear()
        edits.update(entry["edits"])
        board = dict(base)
        board.update(edits)
        prediction = {
            "px": entry["px"], "py": entry["py"],
            "g": entry["g"], "exact": True
        }
        plan = []

    if prediction is None:
        py = memory.get("py", 0)
        gravity = memory.get("g", -1 if player_top >= 31 else 1)
    elif prediction.get("reset"):
        gravity = -1
        if board:
            bounds = [y for x, y in board]
            py = _align(
                cells, board, range(min(bounds) - 12, max(bounds) + 13),
                memory.get("initial_py", 0), px
            )
        else:
            py = None
        if py is None:
            py = 0
            base.clear()
            edits.clear()
            board.clear()
        plan = []
    else:
        gravity = prediction["g"]
        if prediction["exact"]:
            py = prediction["py"]
            if px != prediction.get("px", px):
                plan = []
        else:
            boundary = prediction["py"]
            py = _align(
                cells, board,
                [boundary + gravity * d for d in range(100)],
                boundary + gravity * 3, px
            )
            if py is None:
                py = boundary + gravity * 12
            plan = []

    memory.setdefault("initial_py", py)
    unexpected = False
    uncertain = memory.pop("uncertain_growth", set())

    for (x, ry), value in cells.items():
        key = (x, py + ry)
        if x == px and ry == 0 and key in board:
            value = board[key]

        old = board.get(key)
        if old is not None and old != value:
            unexpected = True

        if key in edits:
            if edits[key] != value:
                if key in uncertain:
                    base[key] = value
                    edits.pop(key, None)
                else:
                    edits[key] = value
            if base.get(key) == value:
                edits.pop(key, None)
        else:
            base[key] = value
        board[key] = value

    if unexpected:
        plan = []

    memory["py"] = py
    memory["g"] = gravity
    position = px, py, gravity
    if position != memory.get("last_position"):
        visits[position] = visits.get(position, 0) + 1
    memory["last_position"] = position

    state_key = _key(px, py, gravity, edits)
    forbidden = banned.get(state_key, set())
    model = _model(board, px, py, gravity, player_top, available)
    root = model["root"]
    tile = model["tile"]
    transition = model["transition"]

    action = None
    mode = "plan"
    if plan:
        candidate = plan[0]
        nxt, status = transition(root, candidate)
        if candidate not in forbidden and nxt is not None and status != 3:
            action = candidate
            plan = plan[1:]
        else:
            plan = []

    if action is None:
        route, mode = _search(model, visits, forbidden, deadline)

        # Once a growth puzzle's goal and gravity switch are both known,
        # a restart can be much cheaper than undoing an exploratory tunnel.
        # Keep the original terrain, but restore all clicked tiles.
        if (model["strategic"] and mode != "win"
                and not memory.get("knowledge_restart")
                and "RESET" in available):
            memory["knowledge_restart"] = True
            edits.clear()
            history.clear()
            visits.clear()
            banned.clear()
            counts.clear()
            memory["prediction"] = {"reset": True}
            memory["last_picture"] = obs[:63].copy()
            return "RESET", None, None

        if route:
            action = route[0]
            plan = route[1:]

    if action is None:
        if history and "ACTION7" in available:
            entry = history.pop()
            banned.setdefault(entry["key"], set()).add(entry["action"])
            edits.clear()
            edits.update(entry["edits"])
            memory["prediction"] = {
                "px": entry["px"], "py": entry["py"],
                "g": entry["g"], "exact": True
            }
            memory["last_picture"] = obs[:63].copy()
            return "ACTION7", None, None

        if "RESET" in available and memory.get("resets", 0) < 1:
            memory["resets"] = memory.get("resets", 0) + 1
            edits.clear()
            history.clear()
            visits.clear()
            memory["prediction"] = {"reset": True}
            memory["last_picture"] = obs[:63].copy()
            return "RESET", None, None

        choices = []
        for candidate in model["actions"](root):
            if candidate in forbidden:
                continue
            nxt, status = transition(root, candidate)
            if nxt is None or status == 3:
                continue
            score = (
                10 * (status == 1) + 2 * (status == 2)
                + abs(nxt[1] - py)
                - visits.get(nxt[:3], 0)
                + 0.1 * (candidate[0] == "c")
            )
            choices.append((score, candidate))

        if choices:
            action = max(choices, key=lambda item: item[0])[1]
        elif "RESET" in available:
            memory.clear()
            return "RESET", None, None
        else:
            name = next((a for a in ("ACTION7", "ACTION3", "ACTION4")
                         if a in available), next(iter(available), "ACTION3"))
            return name, None, None

    edge = state_key, action
    counts[edge] = counts.get(edge, 0) + 1
    if counts[edge] >= 2:
        banned.setdefault(state_key, set()).add(action)

    history.append({
        "px": px, "py": py, "g": gravity,
        "edits": dict(edits), "key": state_key, "action": action,
    })

    nxt, status = transition(root, action)
    if nxt is not None:
        _update_edits(memory, model["changes"](nxt))

        if action[0] == "c" and tile(root, action[1], action[2]) == GROW:
            tx, ty = action[1], action[2]
            uncertain = set()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                key = tx + dx, ty + dy
                if 0 <= key[0] < 11 and key != (px, py) and key not in board:
                    base[key] = EMPTY
                    edits[key] = GROW
                    uncertain.add(key)
            memory["uncertain_growth"] = uncertain

        memory["prediction"] = {
            "px": nxt[0], "py": nxt[1], "g": nxt[2],
            "exact": status != 2, "action": action,
        }
        if status != 2:
            memory["plan"] = plan

    memory["last_picture"] = obs[:63].copy()

    if action[0] == "m":
        return ("ACTION3" if action[1] < 0 else "ACTION4"), None, None

    row = action[2] * 6 - root[6] + 3
    col = action[1] * 6 + 3
    return "ACTION6", int(max(0, min(62, row))), int(max(0, min(63, col)))

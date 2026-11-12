import numpy as np
import heapq
import time

D = (-256, 256, -1, 1)
ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
COLORS = (14, 8, 9)
EMPTY = ((), (), (), (), ())


def pack(x, y):
    return (y + 32) * 256 + x + 32


def unpack(p):
    return p % 256 - 32, p // 256 - 32


def distance(p, q):
    x, y = unpack(p)
    a, b = unpack(q)
    return abs(x - a) + abs(y - b)


def make_patterns():
    arrays, info = [], []
    peg = np.zeros((6, 6), bool)
    peg[1, 2:4] = True
    peg[2:4, 1:5] = True
    peg[4, 2:4] = True

    source = np.zeros((6, 6), bool)
    for y, row in enumerate(
        (" xxxx ", "xx  xx", "x    x", "x    x", "xx  xx", " xxxx ")
    ):
        for x, ch in enumerate(row):
            source[y, x] = ch == "x"

    ring = np.zeros((6, 6), bool)
    ring[1, 2:4] = True
    ring[2:4, 1] = True
    ring[2:4, 4] = True
    ring[4, 2:4] = True
    middle = np.zeros((6, 6), bool)
    middle[2:4, 2:4] = True

    purple = np.full((6, 6), -1, np.int16)
    palette = {"x": 15, "w": 0, "n": 7, "b": 5}
    for y, row in enumerate(
        (".xxxx.", ".xwnx.", ".xnnx.", ".xxxx.", ".bbbb.", "......")
    ):
        for x, ch in enumerate(row):
            if ch in palette:
                purple[y, x] = palette[ch]

    for kind, border, inside in ((1, 0, 1), (2, 11, 12)):
        base = np.full((6, 6), border, np.int16)
        base[1:5, 1:5] = inside
        for entity in (0, 14, 8, 9, 16, 15):
            a = base.copy()
            if entity in (14, 8, 9, 16):
                a[peg] = 2 if entity == 16 else entity
            elif entity == 15:
                mask = purple >= 0
                a[mask] = purple[mask]
            arrays.append(a.ravel())
            info.append((kind, entity, 0))
            if entity in (14, 8, 9, 16):
                b = a.copy()
                b[source] = 3
                arrays.append(b.ravel())
                info.append((kind, entity, 1))
            if entity == 0:
                for mask in (ring, middle):
                    b = a.copy()
                    b[mask] = 2
                    arrays.append(b.ravel())
                    info.append((kind, 0, 2))
    return np.asarray(arrays), info


PATTERNS, INFO = make_patterns()


def read_frame(obs, old_phase):
    best = -1
    phase = old_phase or (0, 0)
    for py in range(6):
        ys = np.arange(py, 59, 6)
        for px in range(6):
            xs = np.arange(px, 59, 6)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            yy, xx = yy.ravel(), xx.ravel()
            blocks = obs[
                yy[:, None, None] + np.arange(6)[None, :, None],
                xx[:, None, None] + np.arange(6)[None, None, :]
            ].reshape(-1, 36)
            mismatch = blocks[:, None, :] != PATTERNS[None, :, :]
            if py == 0:
                mismatch[yy == 0, :, :6] = False
            errors = mismatch.sum(axis=2).min(axis=1)
            weights = np.maximum(0, 6 - errors)
            score = float(np.dot(weights, weights))
            if old_phase == (px, py):
                score += 0.001
            if score > best:
                best, phase = score, (px, py)

    px, py = phase
    cells = []
    for y in range(py - 6, 64, 6):
        for x in range(px - 6, 64, 6):
            xa, xb = max(0, x), min(64, x + 6)
            ya, yb = max(1, y), min(64, y + 6)
            if xb <= xa or yb <= ya:
                continue
            full = x >= 0 and x + 6 <= 64 and y >= 1 and y + 6 <= 64
            kind = entity = selected = 0
            if xb - xa >= 2 and yb - ya >= 2:
                indices = (
                    np.arange(ya - y, yb - y)[:, None] * 6
                    + np.arange(xa - x, xb - x)[None, :]
                ).ravel()
                vals = obs[ya:yb, xa:xb].ravel()
                errors = (PATTERNS[:, indices] != vals).sum(axis=1)
                index = int(errors.argmin())
                if errors[index] <= int(len(indices) * 0.065):
                    kind, entity, selected = INFO[index]
            if not kind:
                if (
                    x + 2 >= 0 and x + 3 < 64
                    and y + 2 >= 1 and y + 3 < 64
                    and np.all(obs[y + 2:y + 4, x + 2:x + 4] == 5)
                ):
                    kind = 3
            cells.append((
                (x - px) // 6, (y - py) // 6,
                kind, entity, selected, full
            ))
    return phase, cells


def objects(state):
    return {
        p: color
        for group, color in zip(state[:3], COLORS)
        for p in group
    }


def find_offset(data, phase, cells):
    if "ox" not in data:
        return phase
    old_x, old_y = data["ox"], data["oy"]
    floor, rails, blank = data["floor"], data["rails"], data["blank"]
    obj = objects(data["state"])
    platforms = set(data["state"][3])
    hurdles = set(data["state"][4])
    bx = round((phase[0] - old_x) / 6)
    by = round((phase[1] - old_y) / 6)
    useful = [c for c in cells if c[2] or c[5]]
    best = None
    for sy in range(by - 12, by + 13):
        for sx in range(bx - 14, bx + 15):
            score = -0.05 * (abs(sx - bx) + abs(sy - by))
            for x, y, kind, entity, selection, full in useful:
                p = pack(x + sx, y + sy)
                if kind == 1:
                    if p in floor:
                        score += 3
                    elif p in rails and p not in platforms:
                        score -= 2
                    elif p in blank:
                        score -= 1
                elif kind == 2:
                    if p in platforms:
                        score += 17
                    elif p in floor:
                        score -= 3
                elif kind == 3:
                    if p in rails:
                        score += 1.6
                    elif p in floor:
                        score -= 2.5
                elif full:
                    if p in floor:
                        score -= 2
                    if p in platforms:
                        score -= 6
                if entity in COLORS:
                    if obj.get(p) == entity:
                        score += 24
                    elif p in obj:
                        score -= 13
                    elif p in floor or p in platforms:
                        score -= 3
                elif entity == 15:
                    if p in hurdles:
                        score += 10
                elif entity == 0 and full:
                    if p in obj:
                        score -= 12
                    if p in hurdles:
                        score -= 6
            item = (score, -abs(sx - bx) - abs(sy - by), sx, sy)
            if best is None or item > best:
                best = item
    return phase[0] - best[2] * 6, phase[1] - best[3] * 6


def visible_cells(ox, oy):
    xmin = (-ox - 5) // 6
    xmax = (63 - ox) // 6
    ymin = (1 - oy - 5) // 6
    ymax = (63 - oy) // 6
    return {
        pack(x, y)
        for y in range(ymin, ymax + 1)
        for x in range(xmin, xmax + 1)
    }


def click(p, ox, oy):
    x, y = unpack(p)
    left, top = ox + 6 * x, oy + 6 * y
    if left >= 64 or left + 6 <= 0 or top >= 64 or top + 6 <= 1:
        return None
    return (
        "ACTION6",
        int(max(1, min(63, top + 3))),
        int(max(0, min(63, left + 3)))
    )


def count(state):
    return len(state[0]) + len(state[1])


def transition(state, op, rails):
    groups = [set(g) for g in state[:3]]
    platforms, hurdles = set(state[3]), set(state[4])
    if op[0] == "j":
        src, dst, color = op[1:]
        ci = COLORS.index(color)
        if src not in groups[ci]:
            return state
        groups[ci].remove(src)
        groups[ci].add(dst)
        mid = (src + dst) // 2
        if ci < 2:
            groups[ci].discard(mid)
    else:
        delta = D[op[1]]
        tracks = rails | set(op[2])
        for p in sorted(platforms, reverse=delta > 0):
            q = p + delta
            if q in platforms or q not in tracks:
                continue
            platforms.remove(p)
            platforms.add(q)
            for group in groups:
                if p in group:
                    group.remove(p)
                    group.add(q)
            if p in hurdles:
                hurdles.remove(p)
                hurdles.add(q)
    return tuple(tuple(sorted(g)) for g in groups) + (
        tuple(sorted(platforms)), tuple(sorted(hurdles))
    )


def moves(state, data, visible, restrict_blue=False):
    floor, rails = data["floor"], data["rails"]
    platforms = set(state[3])
    nonblue = set(state[0]) | set(state[1])
    jumpers = set().union(*map(set, state[:3]), set(state[4]))
    occupied = jumpers | data["blocked"]
    landing = floor | platforms
    n = count(state)
    result = []
    for group, color in zip(state[:3], COLORS):
        for src in group:
            if src not in visible:
                continue
            if restrict_blue and color == 9 and nonblue:
                if min(distance(src, q) for q in nonblue) > 7:
                    continue
            for delta in D:
                mid, dst = src + delta, src + 2 * delta
                if mid not in jumpers or dst not in landing:
                    continue
                if dst in occupied or dst not in visible:
                    continue
                if dst in data["bad_dest"] and n <= data["bad_dest"][dst]:
                    continue
                op = ("j", src, dst, color)
                if (state, op) in data["banned"]:
                    continue
                nxt = transition(state, op, rails)
                result.append((count(nxt), op, nxt, False))
    result.sort(key=lambda v: v[0])
    for _, op, nxt, frontier in result:
        yield op, nxt, frontier

    for direction, delta in enumerate(D):
        extras = []
        for p in platforms & nonblue:
            q = p + delta
            if (
                q not in rails and q not in floor and q not in data["blank"]
                and q not in visible
                and (p, direction, data["ox"], data["oy"])
                not in data["view_bans"]
            ):
                extras.append(q)
        op = ("t", direction, tuple(sorted(extras)))
        if (state, op) in data["banned"]:
            continue
        nxt = transition(state, op, rails)
        if nxt == state:
            continue
        stopped = False
        for p in platforms & nonblue:
            q = p + delta
            if q in set(nxt[0]) | set(nxt[1]) and q not in visible:
                if (
                    (p, direction, data["ox"], data["oy"])
                    in data["view_bans"]
                ):
                    stopped = True
        if not stopped:
            yield op, nxt, bool(extras)


def geometry(state):
    value = 0
    for group in state[:2]:
        if len(group) > 1:
            for p in group:
                value += min(distance(p, q) for q in group if q != p)
    return value


def solitaire(state, data, visible, deadline):
    if state[2] or state[4] or (state[0] and state[1]):
        return None
    ci = 0 if state[0] else 1
    pieces = state[ci]
    if len(pieces) <= 1:
        return None
    cells = sorted(
        ((data["floor"] | set(state[3])) & visible) - data["blocked"]
    )
    index = {p: i for i, p in enumerate(cells)}
    if any(p not in index for p in pieces):
        return None
    triples = []
    for src in cells:
        for delta in D:
            mid, dst = src + delta, src + 2 * delta
            if mid in index and dst in index:
                triples.append((
                    1 << index[src], 1 << index[mid], 1 << index[dst],
                    src, dst
                ))
    initial = sum(1 << index[p] for p in pieces)
    failed, path = set(), []
    nodes = 0

    def dfs(bits):
        nonlocal nodes
        nodes += 1
        if nodes % 256 == 0 and time.perf_counter() > deadline:
            raise TimeoutError
        if bits & (bits - 1) == 0:
            return True
        if bits in failed:
            return False
        n = bits.bit_count()
        successors = []
        for a, b, c, src, dst in triples:
            if bits & a and bits & b and not bits & c:
                if dst in data["bad_dest"] and n <= data["bad_dest"][dst]:
                    continue
                nxt = bits ^ (a | b | c)
                positions = [
                    cells[i] for i in range(len(cells)) if nxt >> i & 1
                ]
                isolation = spread = 0
                if len(positions) > 1:
                    for p in positions:
                        nearest = min(
                            distance(p, q) for q in positions if q != p
                        )
                        isolation += nearest > 2
                        spread += nearest
                successors.append((isolation, spread, nxt, src, dst))
        successors.sort()
        for _, _, nxt, src, dst in successors:
            path.append(("j", src, dst, COLORS[ci]))
            if dfs(nxt):
                return True
            path.pop()
        failed.add(bits)
        return False

    try:
        if dfs(initial):
            current = state
            for op in path:
                if (current, op) in data["banned"]:
                    return None
                current = transition(current, op, data["rails"])
            return path.copy()
    except TimeoutError:
        pass
    return None


def blue_helpers(state, data, visible, deadline):
    """Move blue pegs to enable an unvisited colored-peg landing."""
    platforms = set(state[3])
    occupied = set().union(*map(set, state[:3]), set(state[4]))
    occupied |= data["blocked"]
    landings = data["floor"] | platforms
    active_blues = [p for p in state[2] if p in visible]
    if not active_blues:
        return None

    tasks = []
    immediate = []
    for group, color in zip(state[:2], COLORS[:2]):
        for src in group:
            if src not in visible or src in platforms:
                continue
            for delta in D:
                mid, dst = src + delta, src + 2 * delta
                if dst not in visible or dst not in landings:
                    continue
                if dst in occupied or (color, dst) in data["landed"]:
                    continue
                if (
                    dst in data["bad_dest"]
                    and count(state) <= data["bad_dest"][dst]
                ):
                    continue
                op = ("j", src, dst, color)
                if (state, op) in data["banned"]:
                    continue
                if mid in occupied - data["blocked"]:
                    immediate.append(op)
                elif mid in landings and mid in visible:
                    mx, my = unpack(mid)
                    compatible = [
                        b for b in active_blues
                        if (unpack(b)[0] - mx) % 2 == 0
                        and (unpack(b)[1] - my) % 2 == 0
                    ]
                    if compatible:
                        tasks.append((mid, op, tuple(compatible)))

    if immediate:
        immediate.sort(
            key=lambda op: (-unpack(op[2])[0], distance(op[1], op[2]))
        )
        return [immediate[0]]
    if not tasks:
        return None

    max_x = max(unpack(op[2])[0] for mid, op, compatible in tasks)

    def heuristic(s):
        blue = s[2]
        best = 1000
        for mid, op, compatible in tasks:
            mx, my = unpack(mid)
            ds = [
                distance(b, mid) for b in blue if b in visible
                and (unpack(b)[0] - mx) % 2 == 0
                and (unpack(b)[1] - my) % 2 == 0
            ]
            if ds:
                value = (
                    2.4 * min(ds)
                    + 0.35 * (max_x - unpack(op[2])[0])
                )
                best = min(best, value)
        return best

    costs = {state: 0}
    parent = {state: None}
    heap = [(heuristic(state), 0, 0, state)]
    serial = examined = 0

    def path_to(s):
        path = []
        while parent[s] is not None:
            prev, op = parent[s]
            path.append(op)
            s = prev
        return path[::-1]

    while heap:
        _, cost, _, current = heapq.heappop(heap)
        if costs.get(current) != cost:
            continue
        examined += 1
        if examined % 64 == 0:
            if time.perf_counter() >= deadline or examined > 28000:
                break

        current_blue = set(current[2])
        enabled = []
        for mid, op, compatible in tasks:
            if mid in current_blue and (current, op) not in data["banned"]:
                enabled.append(op)
        if enabled:
            enabled.sort(key=lambda op: -unpack(op[2])[0])
            return path_to(current) + [enabled[0]]

        for op, nxt, frontier in moves(current, data, visible, False):
            if op[0] != "j" or op[3] != 9:
                continue
            new_cost = cost + 2
            if new_cost >= costs.get(nxt, 10 ** 9):
                continue
            if len(costs) > 80000:
                continue
            costs[nxt] = new_cost
            parent[nxt] = current, op
            serial += 1
            repeat = data["edges"].get((current, op), 0)
            heapq.heappush(
                heap,
                (new_cost + heuristic(nxt) + min(8, repeat * 2),
                 new_cost, serial, nxt)
            )
    return None


def search(state, data, visible):
    started = time.perf_counter()
    deadline = started + 5.8
    simple = solitaire(state, data, visible, started + 0.9)
    if simple:
        return simple

    n0 = count(state)
    goal = max(1, int(bool(state[0])) + int(bool(state[1])))
    landed, visits = data["landed"], data["visited"]
    costs = {state: 0}
    parents = {state: None}
    serial = 0
    heap = [(35 * (n0 - goal) + 2.5 * geometry(state), 0, 0, state)]
    best_partial = best_frontier = best_novel = None
    examined = 0
    if state[2]:
        primary_duration = 1.25 if n0 > goal else 0.45
    else:
        primary_duration = 3.0 if n0 > goal else 0.8
    primary_end = min(deadline, time.perf_counter() + primary_duration)

    def path_to(s):
        path = []
        while parents[s] is not None:
            prev, op = parents[s]
            path.append(op)
            s = prev
        return path[::-1]

    while heap:
        _, g, _, current = heapq.heappop(heap)
        if costs.get(current) != g:
            continue
        examined += 1
        if examined % 128 == 0:
            if time.perf_counter() >= primary_end or examined > 60000:
                break
        n = count(current)
        if n <= goal and n < n0:
            return path_to(current)
        successors = list(moves(current, data, visible, False))
        if n < n0 and successors:
            rank = (n, g + geometry(current) * 0.7)
            if best_partial is None or rank < best_partial[0]:
                best_partial = rank, current
        for op, nxt, frontier in successors:
            ng = g + (2 if op[0] == "j" else 1)
            if frontier:
                destination_x = max(unpack(p)[0] for p in op[2])
                rank = (count(nxt), -destination_x, ng)
                if best_frontier is None or rank < best_frontier[0]:
                    best_frontier = rank, current, op
                continue
            if op[0] == "j" and op[3] != 9:
                key = op[3], op[2]
                if key not in landed:
                    x, y = unpack(op[2])
                    rank = (count(nxt), -x, ng + 2 * visits.get(nxt, 0))
                    if best_novel is None or rank < best_novel[0]:
                        best_novel = rank, current, op
            if ng >= costs.get(nxt, 10 ** 9) or len(costs) > 140000:
                continue
            costs[nxt] = ng
            parents[nxt] = current, op
            serial += 1
            h = 35 * max(0, count(nxt) - goal) + 2.5 * geometry(nxt)
            heapq.heappush(
                heap, (ng + h + min(15, 3 * visits.get(nxt, 0)),
                       ng, serial, nxt)
            )

    if best_partial is not None:
        return path_to(best_partial[1])

    if state[2]:
        helper = blue_helpers(
            state, data, visible,
            min(deadline, time.perf_counter() + 1.65)
        )
        if helper:
            return helper
        if best_frontier is not None:
            return path_to(best_frontier[1]) + [best_frontier[2]]

    candidate_cells = [
        p for p in data["floor"] & visible
        if p not in data["blocked"]
        and any((color, p) not in landed for color in COLORS[:2])
    ]
    targets = []
    if candidate_cells:
        candidate_cells.sort(
            key=lambda p: (
                -unpack(p)[0],
                min((distance(p, q) for q in state[0] + state[1]), default=0)
            )
        )
        xmax = unpack(candidate_cells[0])[0]
        targets = [
            p for p in candidate_cells if unpack(p)[0] >= xmax - 1
        ][:10]

    def explore_h(s):
        pegs = s[0] + s[1]
        if not pegs or not targets:
            return 0
        nearest = min(distance(p, t) for p in pegs for t in targets)
        cooperation = 0
        if len(pegs) > 1:
            cooperation = sum(
                min(14, min(distance(p, t) for t in targets))
                for p in pegs
            )
        return 3.5 * nearest + 0.7 * cooperation

    ecost = {state: 0}
    eparent = {state: None}
    eheap = [(explore_h(state), 0, 0, state)]
    explored = 0
    novel_records, frontier_records = [], []

    def epath(s):
        result = []
        while eparent[s] is not None:
            prev, op = eparent[s]
            result.append(op)
            s = prev
        return result[::-1]

    while eheap and time.perf_counter() < deadline:
        _, g, _, current = heapq.heappop(eheap)
        if ecost.get(current) != g:
            continue
        explored += 1
        if explored > 65000:
            break
        for op, nxt, frontier in moves(
            current, data, visible, not bool(state[2])
        ):
            ng = g + (2 if op[0] == "j" else 1)
            if count(nxt) < n0:
                return epath(current) + [op]
            if frontier:
                x = max(unpack(p)[0] for p in op[2])
                frontier_records.append(
                    ((count(nxt), -x, ng), current, op)
                )
                continue
            if op[0] == "j" and op[3] != 9:
                key = op[3], op[2]
                if key not in landed:
                    x, y = unpack(op[2])
                    rank = (
                        count(nxt), -x,
                        ng + 3 * visits.get(nxt, 0)
                    )
                    novel_records.append((rank, current, op))
                    if op[2] in targets and visits.get(nxt, 0) == 0:
                        return epath(current) + [op]
            if ng >= ecost.get(nxt, 10 ** 9) or len(ecost) > 150000:
                continue
            ecost[nxt] = ng
            eparent[nxt] = current, op
            serial += 1
            heapq.heappush(
                eheap,
                (ng + explore_h(nxt) + min(18, 4 * visits.get(nxt, 0)),
                 ng, serial, nxt)
            )

    if frontier_records:
        frontier_records.sort(key=lambda z: z[0])
        _, s, op = frontier_records[0]
        return epath(s) + [op]
    if best_frontier is not None:
        return path_to(best_frontier[1]) + [best_frontier[2]]
    if novel_records:
        novel_records.sort(key=lambda z: z[0])
        _, s, op = novel_records[0]
        return epath(s) + [op]
    if best_novel is not None:
        return path_to(best_novel[1]) + [best_novel[2]]

    choices = []
    for op, nxt, frontier in moves(state, data, visible, False):
        repeat = data["edges"].get((state, op), 0)
        rank = (
            count(nxt), repeat, visits.get(nxt, 0),
            geometry(nxt), 0 if op[0] == "j" else 1
        )
        choices.append((rank, op))
    if choices:
        choices.sort(key=lambda z: z[0])
        return [choices[0][1]]
    return []


def undo(data, available):
    data["plan"] = []
    data["pending"] = None
    data["last"] = None
    if "ACTION7" in available:
        return "ACTION7", None, None
    return "RESET", None, None


def execute(data, op, state):
    data["last"] = {"before": state, "op": op, "checked": False}
    data["state"] = transition(state, op, data["rails"])
    edge = state, op
    data["edges"][edge] = data["edges"].get(edge, 0) + 1
    if op[0] == "j":
        data["landed"].add((op[3], op[2]))


def act(obs, available, memory):
    available = {str(a).split(".")[-1].upper() for a in available}
    data = memory.setdefault("player", {})
    if "floor" not in data:
        data.update({
            "floor": set(), "rails": set(), "blank": set(),
            "blocked": set(), "state": EMPTY, "plan": [],
            "pending": None, "last": None, "bad_dest": {},
            "banned": set(), "view_bans": set(), "visited": {},
            "landed": set(), "edges": {}
        })

    phase, local = read_frame(obs, data.get("phase"))
    old_offset = data.get("ox"), data.get("oy")
    last = data.get("last")

    if (
        last and not last["checked"] and last["op"][0] == "t"
        and "image" in data
        and np.array_equal(obs[1:], data["image"][1:])
    ):
        data["banned"].add((last["before"], last["op"]))
        data["state"] = last["before"]
        data["plan"] = []
        for p in last["op"][2]:
            data["blank"].add(p)
        last["checked"] = True

    ox, oy = find_offset(data, phase, local)
    sx, sy = (phase[0] - ox) // 6, (phase[1] - oy) // 6
    cells = [
        (pack(x + sx, y + sy), kind, entity, selected, full)
        for x, y, kind, entity, selected, full in local
    ]
    gray = {p for p, k, e, s, f in cells if e == 16}
    if gray:
        if last and last["op"][0] == "j":
            dst = last["op"][2]
            data["bad_dest"][dst] = max(
                data["bad_dest"].get(dst, 0), count(last["before"])
            )
            data["state"] = last["before"]
        elif last:
            data["banned"].add((last["before"], last["op"]))
            data["state"] = last["before"]
        else:
            data["initial_gray"] = gray
        data["image"] = obs.copy()
        return undo(data, available)

    predicted = data["state"]
    groups = [set(g) for g in predicted[:3]]
    platforms, hurdles = set(predicted[3]), set(predicted[4])
    full_seen = {p for p, k, e, s, full in cells if full}
    for group in groups:
        group.difference_update(full_seen)
    platforms.difference_update(full_seen)
    hurdles.difference_update(full_seen)

    selected = None
    has_source = any(s == 1 for p, k, e, s, f in cells)
    old_sizes = len(data["floor"]), len(data["rails"])

    for p, kind, entity, selection, full in cells:
        if kind == 1:
            data["floor"].add(p)
            data["blank"].discard(p)
        elif kind == 2:
            platforms.add(p)
            data["rails"].add(p)
            data["blank"].discard(p)
        elif kind == 3:
            data["rails"].add(p)
            data["blank"].discard(p)
        elif full:
            data["blank"].add(p)

        if entity in COLORS:
            groups[COLORS.index(entity)].add(p)
            data["landed"].add((entity, p))
        elif entity == 15:
            hurdles.add(p)

        if selection == 1:
            selected = p
        elif selection == 2:
            if not has_source and data.get("pending") is None:
                data["blocked"].add(p)
        elif full:
            data["blocked"].discard(p)

    state = tuple(tuple(sorted(g)) for g in groups) + (
        tuple(sorted(platforms)), tuple(sorted(hurdles))
    )
    if "initial_gray" in data:
        old = data.pop("initial_gray")
        now = set().union(*map(set, state[:3]))
        for p in old - now:
            data["bad_dest"][p] = max(
                data["bad_dest"].get(p, 0), count(state)
            )

    data["phase"] = phase
    data["ox"], data["oy"] = ox, oy
    data["state"] = state
    data["image"] = obs.copy()
    visible = visible_cells(ox, oy)

    if last and not last["checked"] and last["op"][0] == "t":
        if old_offset == (ox, oy):
            direction = last["op"][1]
            delta = D[direction]
            before = last["before"]
            carried = (set(before[0]) | set(before[1])) & set(before[3])
            exits = [
                p for p in carried
                if p + delta not in visible
                and p + delta in set(state[0]) | set(state[1])
            ]
            if exits:
                for p in exits:
                    data["view_bans"].add((p, direction, ox, oy))
                data["state"] = before
                return undo(data, available)

    if last and not last["checked"]:
        last["checked"] = True
        if state == last["before"] and last["op"][0] == "j":
            data["banned"].add((last["before"], last["op"]))
            data["plan"] = []

    new_information = old_sizes != (
        len(data["floor"]), len(data["rails"])
    )
    if state != predicted or new_information or old_offset != (ox, oy):
        data["plan"] = []
        if data.get("pending") and state != data["pending"][1]:
            data["pending"] = None

    pending = data.get("pending")
    if pending:
        op, before = pending
        target = click(op[2], ox, oy)
        if selected == op[1] and target is not None:
            data["pending"] = None
            if data["plan"] and data["plan"][0] == op:
                data["plan"].pop(0)
            execute(data, op, state)
            return target
        data["pending"] = None
        data["plan"] = []
        data["banned"].add((before, op))

    data["visited"][state] = data["visited"].get(state, 0) + 1
    valid = {op for op, nxt, frontier in moves(state, data, visible)}
    if data["plan"] and data["plan"][0] not in valid:
        data["plan"] = []
    if not data["plan"]:
        data["plan"] = search(state, data, visible)

    if data["plan"]:
        op = data["plan"][0]
        if op[0] == "j":
            source = click(op[1], ox, oy)
            target = click(op[2], ox, oy)
            if source is not None and target is not None:
                if selected == op[1]:
                    data["plan"].pop(0)
                    execute(data, op, state)
                    return target
                data["pending"] = op, state
                return source
        else:
            action = ACTIONS[op[1]]
            if action in available:
                data["plan"].pop(0)
                execute(data, op, state)
                return action, None, None

    last = data.get("last")
    if last:
        data["banned"].add((last["before"], last["op"]))
        data["state"] = last["before"]
    if "ACTION7" in available:
        return undo(data, available)
    if "RESET" in available:
        return "RESET", None, None
    for action in ACTIONS:
        if action in available:
            return action, None, None
    return "ACTION6", 32, 32

import heapq
import itertools
from collections import deque
from time import perf_counter

import numpy as np


_DIRS = ((0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4))


def _make_templates():
    peg = ["      ", "  oo  ", " oooo ",
           " oooo ", "  oo  ", "      "]
    ring = [" xxxx ", "xx  xx", "x    x",
            "x    x", "xx  xx", " xxxx "]
    arrow = ["      ", "  oo  ", " o  o ",
             " o  o ", "  oo  ", "      "]
    purple = [" ffff ", " f07f ", " f77f ",
              " ffff ", " 5555 ", "      "]
    result = []
    for kind, border, inside in (("w", 0, 1), ("p", 11, 12)):
        base = np.full((6, 6), border, dtype=np.int8)
        base[1:5, 1:5] = inside
        result.append((base.copy(), kind, 0, False, False))
        for color, occupant in ((14, 1), (8, 2), (9, 3),
                                (2, 1), (4, 1)):
            a = base.copy()
            for y, row in enumerate(peg):
                for x, c in enumerate(row):
                    if c == "o":
                        a[y, x] = color
            result.append((a.copy(), kind, occupant, False, False))
            for y, row in enumerate(ring):
                for x, c in enumerate(row):
                    if c == "x":
                        a[y, x] = 3
            result.append((a, kind, occupant, True, False))
        a = base.copy()
        for y, row in enumerate(purple):
            for x, c in enumerate(row):
                if c != " ":
                    a[y, x] = int(c, 16)
        result.append((a, kind, 4, False, False))
        a = base.copy()
        for y, row in enumerate(arrow):
            for x, c in enumerate(row):
                if c == "o":
                    a[y, x] = 2
        result.append((a, kind, 0, False, True))
        a = base.copy()
        a[2:4, 2:4] = 2
        result.append((a, kind, 0, False, True))
    return result


_TEMPLATES = _make_templates()


def _visible(p, ax, ay):
    x, y = ax + 6*p[0], ay + 6*p[1]
    return x < 64 and x+6 > 0 and y < 64 and y+6 > 1


def _full(p, ax, ay):
    x, y = ax + 6*p[0], ay + 6*p[1]
    return 0 <= x <= 58 and 1 <= y <= 58


def _point(p, ax, ay):
    if not _visible(p, ax, ay):
        return None
    return (
        int(max(1, min(63, ay+6*p[1]+3))),
        int(max(0, min(63, ax+6*p[0]+3)))
    )


def _parse(frame, preferred=None):
    a = np.asarray(frame, dtype=np.int8)
    windows = np.lib.stride_tricks.sliding_window_view(a, (6, 6))
    votes = np.zeros((6, 6), dtype=np.int32)
    hits = []
    for i, (template, kind, occupant, selected, arrow) in enumerate(_TEMPLATES):
        match = np.all(windows == template, axis=(-2, -1))
        match[0, :] = False
        ys, xs = np.nonzero(match)
        for y, x in zip(ys.tolist(), xs.tolist()):
            votes[y % 6, x % 6] += 1
            hits.append((x, y, i))

    if votes.max():
        choices = np.argwhere(votes == votes.max())
        if preferred is None:
            py, px = choices[0]
        else:
            ox, oy = preferred

            def distance(q):
                y, x = map(int, q)
                return (
                    min((x-ox) % 6, (ox-x) % 6)
                    + min((y-oy) % 6, (oy-y) % 6)
                )

            py, px = min(choices, key=distance)
        px, py = int(px), int(py)
    elif preferred is not None:
        px, py = preferred[0] % 6, preferred[1] % 6
    else:
        px = py = 5

    cells = {}
    for x, y, i in hits:
        if x % 6 != px or y % 6 != py:
            continue
        _, kind, occupant, selected, arrow = _TEMPLATES[i]
        p = (x-px)//6, (y-py)//6
        value = kind, occupant, selected, arrow
        if p not in cells or occupant or selected or arrow:
            cells[p] = value

    for gy in range(-1, 11):
        y = py + 6*gy
        for gx in range(-1, 11):
            x = px + 6*gx
            p = gx, gy
            if p in cells or (0 <= x <= 58 and 1 <= y <= 58):
                continue
            x0, x1 = max(0, x), min(64, x+6)
            y0, y1 = max(1, y), min(64, y+6)
            if x1 <= x0 or y1 <= y0 or (x1-x0)*(y1-y0) < 6:
                continue
            piece = a[y0:y1, x0:x1]
            found = []
            for t, kind, occupant, selected, arrow in _TEMPLATES:
                if np.array_equal(piece, t[y0-y:y1-y, x0-x:x1-x]):
                    found.append((kind, occupant, selected, arrow))
            if found and len({v[0] for v in found}) == 1:
                occupants = {v[1] for v in found}
                occupant = next(iter(occupants)) if len(occupants) == 1 else None
                cells[p] = (
                    found[0][0], occupant,
                    all(v[2] for v in found),
                    all(v[3] for v in found)
                )

    for gy in range(-1, 11):
        y = py + 6*gy
        for gx in range(-1, 11):
            x = px + 6*gx
            p = gx, gy
            if p in cells:
                continue
            if 0 <= x+2 and x+4 <= 64 and 1 <= y+2 and y+4 <= 64:
                if np.all(a[y+2:y+4, x+2:x+4] == 5):
                    cells[p] = ("r", 0, False, False)

    # Recover rails whose centers are clipped by the viewport.
    for _ in range(2):
        for gy in range(-1, 11):
            y = py + 6*gy
            for gx in range(-1, 11):
                x = px + 6*gx
                p = gx, gy
                if p in cells:
                    continue
                x0, x1 = max(0, x), min(64, x+6)
                y0, y1 = max(1, y), min(64, y+6)
                if x1 <= x0 or y1 <= y0:
                    continue
                horizontal = (
                    (x < 0 or x+6 > 64)
                    and any(
                        cells.get((gx+dx, gy), ("",))[0] in ("r", "p")
                        for dx in (-1, 1)
                    )
                )
                vertical = (
                    (y < 1 or y+6 > 64)
                    and any(
                        cells.get((gx, gy+dy), ("",))[0] in ("r", "p")
                        for dy in (-1, 1)
                    )
                )
                if horizontal and 1 <= y+2 and y+4 <= 64:
                    strip = a[y+2:y+4, x0:x1]
                    if strip.size and np.all(strip == 5):
                        cells[p] = ("r", 0, False, False)
                elif vertical and 0 <= x+2 and x+4 <= 64:
                    strip = a[y0:y1, x+2:x+4]
                    if strip.size and np.all(strip == 5):
                        cells[p] = ("r", 0, False, False)

    empty = set()
    for gy in range(-1, 11):
        y = py + 6*gy
        for gx in range(-1, 11):
            x = px + 6*gx
            if 0 <= x <= 58 and 1 <= y <= 58 and (gx, gy) not in cells:
                empty.add((gx, gy))
    return px, py, cells, empty


def _signature(objects, pads):
    return (
        tuple(sorted((x, y, c) for (x, y), c in objects.items())),
        tuple(sorted(pads))
    )


def _snapshot(m):
    return m["O"].copy(), list(m["P"]), m["ax"], m["ay"]


def _restore(m, snapshot):
    objects, pads, ax, ay = snapshot
    m["O"] = objects.copy()
    m["P"] = list(pads)
    m["ax"], m["ay"] = ax, ay


def _sync(m, px, py, observed, empty):
    if "S" not in m:
        m.update({
            "S": set(), "R": set(), "V": set(), "O": {}, "P": [],
            "ax": px, "ay": py, "plan": [], "pending": None,
            "history": [], "visits": {}, "bad": set(),
            "blocked": set(), "motion_blocked": set(),
            "hazards": set(), "follow": {}, "ticks": 0
        })
        ax, ay = px, py
    else:
        old_ax, old_ay = m["ax"], m["ay"]
        sites, rails, void = m["S"], m["R"], m["V"]
        objects, pads = m["O"], set(m["P"])
        known = sites | rails
        k0 = int(round((old_ax-px)/6))
        l0 = int(round((old_ay-py)/6))
        best = None
        for l in range(l0-4, l0+5):
            ay0 = py + 6*l
            for k in range(k0-12, k0+13):
                ax0 = px + 6*k
                score = -0.035*abs(ax0-old_ax) - 0.4*abs(ay0-old_ay)
                seen = {}
                seen_pads = set()
                for (x, y), value in observed.items():
                    p = x-k, y-l
                    seen[p] = value
                    kind, occupant, _, _ = value
                    if kind == "w":
                        if p in sites:
                            score += 2
                        elif p in rails or p in void:
                            score -= 2.5
                    else:
                        if p in rails:
                            score += 1.3
                        elif p in sites or p in void:
                            score -= 2.5
                    if kind == "p":
                        seen_pads.add(p)
                        score += 12 if p in pads else (-4 if p in known else 0)
                    if occupant:
                        expected = objects.get(p, 0)
                        if expected == occupant:
                            score += 18
                        elif expected:
                            score -= 14
                        elif p in known:
                            score -= 4
                for p, color in objects.items():
                    value = seen.get(p)
                    if value is not None and value[1] is not None:
                        if value[1] != color:
                            score -= 12
                for p in pads:
                    if _full(p, ax0, ay0) and p not in seen_pads:
                        score -= 9
                for x, y in empty:
                    p = x-k, y-l
                    if p in known:
                        score -= 2
                    elif p in void:
                        score += 0.1
                if best is None or score > best[0]:
                    best = score, ax0, ay0
        _, ax, ay = best

    k, l = (ax-px)//6, (ay-py)//6
    global_observed = {(x-k, y-l): v for (x, y), v in observed.items()}
    global_empty = {(x-k, y-l) for x, y in empty}
    old_count = len(m["S"]) + len(m["R"])
    old_objects, old_pads = m["O"].copy(), list(m["P"])
    old_anchor = m["ax"], m["ay"]

    selected = None
    arrows = set()
    seen_pads = []
    for p, (kind, occupant, is_selected, is_arrow) in global_observed.items():
        m["V"].discard(p)
        if kind == "w":
            m["S"].add(p)
        else:
            m["R"].add(p)
        if kind == "p":
            seen_pads.append(p)
        if occupant is not None:
            if occupant:
                m["O"][p] = occupant
            else:
                m["O"].pop(p, None)
        if is_selected and occupant in (1, 2, 3):
            selected = p
        if is_arrow:
            arrows.add(p)

    m["V"].update(global_empty - m["S"] - m["R"])
    preserved = [p for p in old_pads if not _full(p, ax, ay)]
    m["P"] = sorted(set(preserved + seen_pads))
    m["ax"], m["ay"] = ax, ay

    last = m.pop("last_motion", None)
    if last:
        gx, gy = last["delta"]
        sx, sy = last["anchor"]
        if gx and abs(ax-sx) <= 18:
            m["follow"]["x"] = max(-10, min(0, (ax-sx)/gx))
        if gy and abs(ay-sy) <= 18:
            m["follow"]["y"] = max(-10, min(0, (ay-sy)/gy))

    new_map = old_count != len(m["S"]) + len(m["R"])
    if new_map:
        m["bad"].clear()
    if (
        new_map or old_objects != m["O"]
        or set(old_pads) != set(m["P"])
        or old_anchor != (ax, ay)
    ):
        m["plan"] = []
    return selected, arrows


def _motion(m, action):
    dx, dy = next((dx, dy) for dx, dy, a in _DIRS if a == action)
    pads = list(m["P"])
    order = sorted(
        range(len(pads)),
        key=lambda i: pads[i][0] if dx else pads[i][1],
        reverse=(dx > 0 or dy > 0)
    )
    visible_change = False
    green_delta = 0, 0
    for i in order:
        p = pads[i]
        q = p[0]+dx, p[1]+dy
        if q in m["R"] and q not in pads:
            visible_change |= (
                _visible(p, m["ax"], m["ay"])
                or _visible(q, m["ax"], m["ay"])
            )
            if m["O"].get(p) == 1:
                green_delta = dx, dy
            if p in m["O"]:
                m["O"][q] = m["O"].pop(p)
            pads[i] = q
    m["P"] = sorted(pads)
    m["last_motion"] = {
        "delta": green_delta, "anchor": (m["ax"], m["ay"])
    }
    return visible_change


def _jump(m, source, destination):
    color = m["O"].get(source)
    if color not in (1, 2, 3):
        return
    middle = (
        (source[0]+destination[0])//2,
        (source[1]+destination[1])//2
    )
    if color != 3 and m["O"].get(middle) == color:
        m["O"].pop(middle, None)
    m["O"].pop(source, None)
    m["O"][destination] = color


def _legal(m, source, destination):
    color = m["O"].get(source)
    if color not in (1, 2, 3):
        return False
    dx, dy = destination[0]-source[0], destination[1]-source[1]
    if (dx, dy) not in ((0, -2), (0, 2), (-2, 0), (2, 0)):
        return False
    middle = source[0]+dx//2, source[1]+dy//2
    return (
        middle in m["O"]
        and destination not in m["O"]
        and (destination in m["S"] or destination in m["P"])
    )


def _search(m, deadline):
    sites, rails = m["S"], m["R"]
    objects, pads = m["O"], m["P"]
    positions = sorted(sites | rails | set(objects) | set(pads))
    if not positions:
        return []
    index = {p: i for i, p in enumerate(positions)}
    bits = [1 << i for i in range(len(positions))]
    white = sum(bits[index[p]] for p in sites)
    rail_ids = {index[p] for p in rails}
    colors = [0, 0, 0, 0]
    for p, c in objects.items():
        if 1 <= c <= 4:
            colors[c-1] |= bits[index[p]]
    root = tuple(colors)+(tuple(sorted(index[p] for p in pads)),)

    triples = [[] for _ in positions]
    neighbors = [[None]*4 for _ in positions]
    for i, (x, y) in enumerate(positions):
        for d, (dx, dy, _) in enumerate(_DIRS):
            middle = index.get((x+dx, y+dy))
            destination = index.get((x+2*dx, y+2*dy))
            neighbors[i][d] = middle
            if middle is not None and destination is not None:
                triples[i].append((middle, destination))

    ax, ay = m["ax"], m["ay"]
    on_screen = [_visible(p, ax, ay) for p in positions]
    known = sites | rails
    frontier_white, frontier_rail = [], []
    for p in known:
        x, y = ax+6*p[0], ay+6*p[1]
        outward = []
        if x <= 5:
            outward.append((-1, 0))
        if x >= 52:
            outward.append((1, 0))
        if y <= 6:
            outward.append((0, -1))
        if y >= 52:
            outward.append((0, 1))
        if any(
            (p[0]+dx, p[1]+dy) not in known
            and (p[0]+dx, p[1]+dy) not in m["V"]
            for dx, dy in outward
        ):
            (frontier_white if p in sites else frontier_rail).append(p)

    target_count = max(1, int(bool(root[0]))+int(bool(root[1])))
    root_count = (root[0] | root[1]).bit_count()
    no_follow = abs(m["follow"].get("x", 0)) < 0.5
    relay_exploration = bool(root[1]) and no_follow and root_count <= target_count
    if relay_exploration and frontier_white:
        frontier = frontier_white
    else:
        frontier = frontier_white + frontier_rail

    def iterate(mask):
        while mask:
            b = mask & -mask
            yield b.bit_length()-1
            mask -= b

    def metrics(state):
        primary = state[0] | state[1]
        n = primary.bit_count()
        scouts = list(iterate(state[0] or state[1]))
        distance = min(
            (abs(positions[i][0]-q[0])+abs(positions[i][1]-q[1])
             for i in scouts for q in frontier),
            default=0
        )
        span = 0
        for color in state[:2]:
            ids = list(iterate(color))
            if len(ids) > 1:
                xs = [positions[i][0] for i in ids]
                ys = [positions[i][1] for i in ids]
                span += max(xs)-min(xs)+max(ys)-min(ys)
        pmask = sum(bits[i] for i in state[4])
        passengers = (primary & pmask).bit_count()
        return n, distance, span, passengers

    rn, rd, rs, rp = metrics(root)

    def key_to_state(key):
        description, ps = key
        cc = [0, 0, 0, 0]
        for x, y, color in description:
            i = index.get((x, y))
            if i is None:
                return None
            cc[color-1] |= bits[i]
        ids = []
        for p in ps:
            if p not in index:
                return None
            ids.append(index[p])
        return tuple(cc)+(tuple(sorted(ids)),)

    real_visited = set()
    for key in list(m["visits"])[-300:]:
        state = key_to_state(key)
        if state is not None:
            real_visited.add(state)
    bad = set()
    for key in m["bad"]:
        state = key_to_state(key)
        if state is not None:
            bad.add(state)

    root_signature = _signature(objects, pads)
    forbidden = {
        (index[s], index[t])
        for key, s, t in m["blocked"]
        if key == root_signature and s in index and t in index
    }
    blocked_motion = {
        action for key, action in m["motion_blocked"]
        if key == root_signature
    }

    def jumps(state, affected=None, is_root=False):
        occupied = state[0] | state[1] | state[2] | state[3]
        support = white | sum(bits[i] for i in state[4])
        count = (state[0] | state[1]).bit_count()
        for color in range(3):
            for source in iterate(state[color]):
                for middle, destination in triples[source]:
                    mb, tb = bits[middle], bits[destination]
                    if not occupied & mb or occupied & tb or not support & tb:
                        continue
                    if (positions[destination], count) in m["hazards"]:
                        continue
                    if affected is not None:
                        if not ((bits[source] | mb | tb) & affected):
                            continue
                    if is_root:
                        if not on_screen[source] or not on_screen[destination]:
                            continue
                        if (source, destination) in forbidden:
                            continue
                    cc = list(state[:4])
                    cc[color] = (cc[color] ^ bits[source]) | tb
                    removed = color != 2 and bool(cc[color] & mb)
                    if removed:
                        cc[color] ^= mb
                    yield tuple(cc)+(state[4],), source, destination, removed

    motion_cache = {}

    def configurations(start):
        if start in motion_cache:
            return motion_cache[start]
        if not start:
            return []
        maximum = 550 if len(start) > 1 else max(80, len(rail_ids)+5)
        queue = deque([(start, ())])
        seen = {start}
        result = []
        while queue and len(result) < maximum:
            configuration, path = queue.popleft()
            result.append((configuration, path))
            if len(path) >= 55:
                continue
            if len(result) % 32 == 0 and perf_counter() >= deadline:
                break
            for d, (dx, dy, action) in enumerate(_DIRS):
                pp = list(configuration)
                order = sorted(
                    range(len(pp)),
                    key=lambda j: positions[pp[j]][0] if dx else positions[pp[j]][1],
                    reverse=(dx > 0 or dy > 0)
                )
                moved = False
                for j in order:
                    q = neighbors[pp[j]][d]
                    if q is not None and q in rail_ids and q not in pp:
                        pp[j] = q
                        moved = True
                q = tuple(pp)
                if moved and q not in seen:
                    seen.add(q)
                    queue.append((q, path+(action,)))
        if len(motion_cache) >= 100:
            motion_cache.clear()
        motion_cache[start] = result
        return result

    def transport(state, configuration):
        old_mask = sum(bits[i] for i in state[4])
        cc = [color & ~old_mask for color in state[:4]]
        for source, destination in zip(state[4], configuration):
            for color in range(4):
                if state[color] & bits[source]:
                    cc[color] |= bits[destination]
        return tuple(cc)+(tuple(sorted(configuration)),)

    costs = {root: 0}
    parents = {}
    counter = itertools.count()
    heap = [(0, next(counter), root, 0)]
    best_state = None
    best_score = -float("inf")
    expansions = 0

    def reconstruct(state):
        edges = []
        while state != root:
            previous, edge = parents[state]
            edges.append(edge)
            state = previous
        result = []
        for edge in reversed(edges):
            result.extend(edge)
        return result

    def consider(state, previous, edge, cost, jumped=False):
        nonlocal best_state, best_score
        if state == root or state in bad or cost >= costs.get(state, 10**9):
            return False
        if cost > 210:
            return False
        costs[state] = cost
        parents[state] = previous, edge
        n, distance, span, passengers = metrics(state)
        score = (
            200*(rn-n)+3*(rd-distance)+1.5*(rs-span)
            +12*(passengers-rp)-0.12*cost
        )
        if relay_exploration:
            score += 3*(rd-distance)
        if state in real_visited:
            score -= 65
        if score > best_score:
            best_score, best_state = score, state

        if n <= target_count and root_count > target_count:
            return True
        if (
            frontier and n <= target_count and distance <= 1
            and distance < rd and state not in real_visited
            and (not relay_exploration or jumped)
        ):
            return True
        priority = (
            8*n+0.19*cost+0.085*span
            +(0.24 if relay_exploration else 0.12)*distance
            +(2.5 if state in real_visited else 0)
        )
        heapq.heappush(heap, (priority, next(counter), state, cost))
        return False

    while heap and expansions < 14000 and perf_counter() < deadline:
        _, _, state, cost = heapq.heappop(heap)
        if cost != costs.get(state):
            continue
        expansions += 1
        for successor, source, destination, removed in jumps(
            state, is_root=(state == root)
        ):
            edge = [("j", positions[source], positions[destination])]
            if consider(successor, state, edge, cost+2, True):
                return reconstruct(successor)

        if not state[4] or perf_counter() >= deadline:
            continue

        options = []
        deduplicated = set()
        for number, (configuration, path) in enumerate(configurations(state[4])):
            if not path:
                continue
            if number % 24 == 0 and perf_counter() >= deadline:
                break
            if state == root and path[0] in blocked_motion:
                continue
            moved_state = transport(state, configuration)
            affected = sum(bits[i] for i in configuration)

            if (moved_state[0] | moved_state[1]) & affected:
                n, distance, span, passengers = metrics(moved_state)
                if distance < rd or span < rs or frontier:
                    edge = [("m", action) for action in path]
                    if consider(moved_state, state, edge, cost+len(path)):
                        return reconstruct(moved_state)

            for successor, source, destination, removed in jumps(
                moved_state, affected=affected
            ):
                if successor in deduplicated:
                    continue
                deduplicated.add(successor)
                n, distance, span, passengers = metrics(successor)
                rank = (-int(removed), len(path)+0.15*span+0.1*distance)
                options.append((rank, successor, source, destination, path))

        options.sort(key=lambda item: item[0])
        for _, successor, source, destination, path in options[:110]:
            edge = [("m", action) for action in path]
            edge.append(("j", positions[source], positions[destination]))
            if consider(successor, state, edge, cost+len(path)+2, True):
                return reconstruct(successor)

        if len(heap) > 70000:
            heap = heapq.nsmallest(35000, heap)
            heapq.heapify(heap)

    return reconstruct(best_state) if best_state is not None else []


def act(obs, available, memory):
    deadline = perf_counter()+7.35
    available = set(available)
    frame = np.asarray(obs, dtype=np.int8)
    frame_key = frame[1:].tobytes()

    # A failed jump greys the pegs and displays the purple restart icon.
    failed_animation = (
        np.count_nonzero(frame[51:61, 2:15] == 15) >= 8
        and np.count_nonzero(frame[1:] == 2) >= 10
    )
    last_jump = memory.get("last_jump")
    if failed_animation and last_jump and "ACTION7" in available:
        before, source, destination = last_jump
        objects, pads, _, _ = before
        key = _signature(objects, pads)
        count = sum(c in (1, 2) for c in objects.values())
        memory["blocked"].add((key, source, destination))
        memory["hazards"].add((destination, count))
        if memory["history"] and memory["history"][-1][0] == "j":
            memory["history"].pop()
        _restore(memory, before)
        memory["plan"] = []
        memory["pending"] = None
        memory["undo_left"] = 1
        memory.pop("last_jump", None)
        memory.pop("last_motion", None)
        return "ACTION7", None, None

    check = memory.pop("motion_check", None)
    if check is not None and check["visible"] and check["frame"] == frame_key:
        memory["motion_blocked"].add((check["key"], check["action"]))
        _restore(memory, check["snapshot"])
        memory.pop("last_motion", None)
        memory["plan"] = []
        if memory["history"] and memory["history"][-1][0] == "m":
            memory["history"].pop()

    preferred = None
    if "ax" in memory:
        preferred = memory["ax"] % 6, memory["ay"] % 6
    px, py, local, empty = _parse(frame, preferred)
    selected, arrows = _sync(memory, px, py, local, empty)
    memory["ticks"] += 1

    def click(p):
        point = _point(p, memory["ax"], memory["ay"])
        if point is None or "ACTION6" not in available:
            return None
        return "ACTION6", point[0], point[1]

    def execute_jump(source, destination):
        result = click(destination)
        if result is not None:
            before = _snapshot(memory)
            memory["history"].append(("j", before))
            memory["last_jump"] = before, source, destination
            _jump(memory, source, destination)
            memory["pending"] = None
        return result

    if memory.get("undo_left", 0) > 0 and "ACTION7" in available:
        memory["undo_left"] -= 1
        memory["pending"] = None
        memory["plan"] = []
        return "ACTION7", None, None

    pending = memory.get("pending")
    if pending:
        source, destination = pending["source"], pending["destination"]
        source_selected = selected == source
        if not _full(source, memory["ax"], memory["ay"]):
            source_selected |= pending.get("tries", 0) == 0
        clipped_destination = not _full(
            destination, memory["ax"], memory["ay"]
        )
        if (
            (destination in arrows or (source_selected and clipped_destination))
            and _visible(destination, memory["ax"], memory["ay"])
            and _legal(memory, source, destination)
        ):
            result = execute_jump(source, destination)
            if result is not None:
                return result
        if selected == source or pending.get("tries", 0) >= 1:
            key = _signature(memory["O"], memory["P"])
            memory["blocked"].add((key, source, destination))
            memory["pending"] = None
            memory["plan"] = []
        elif (
            _legal(memory, source, destination)
            and _visible(source, memory["ax"], memory["ay"])
        ):
            result = click(source)
            if result is not None:
                pending["tries"] = pending.get("tries", 0)+1
                return result
        else:
            memory["pending"] = None
            memory["plan"] = []

    key = _signature(memory["O"], memory["P"])
    memory["visits"][key] = memory["visits"].get(key, 0)+1

    for _ in range(2):
        if not memory["plan"]:
            memory["plan"] = _search(memory, deadline)
        if not memory["plan"]:
            break
        instruction = memory["plan"].pop(0)
        if instruction[0] == "m":
            action = instruction[1]
            name = "ACTION"+str(action)
            if (
                name not in available
                or (key, action) in memory["motion_blocked"]
            ):
                memory["plan"] = []
                continue
            before = _snapshot(memory)
            visible_change = _motion(memory, action)
            if before[0] == memory["O"] and before[1] == memory["P"]:
                memory["plan"] = []
                memory.pop("last_motion", None)
                continue
            memory["history"].append(("m", before))
            memory["motion_check"] = {
                "snapshot": before, "key": key, "action": action,
                "visible": visible_change, "frame": frame_key
            }
            memory.pop("last_jump", None)
            return name, None, None

        source, destination = instruction[1], instruction[2]
        if (
            not _legal(memory, source, destination)
            or not _visible(source, memory["ax"], memory["ay"])
            or not _visible(destination, memory["ax"], memory["ay"])
        ):
            memory["plan"] = []
            continue
        if selected == source and destination in arrows:
            result = execute_jump(source, destination)
            if result is not None:
                return result
        result = click(source)
        if result is not None:
            memory["pending"] = {
                "source": source, "destination": destination, "tries": 0
            }
            memory.pop("last_jump", None)
            return result

    if memory["history"] and "ACTION7" in available:
        memory["bad"].add(key)
        kind, snapshot = memory["history"].pop()
        _restore(memory, snapshot)
        memory["plan"] = []
        memory["pending"] = None
        memory.pop("last_jump", None)
        memory.pop("last_motion", None)
        memory["undo_left"] = 1 if kind == "j" else 0
        return "ACTION7", None, None

    if selected is not None:
        for destination in sorted(arrows):
            if _legal(memory, selected, destination):
                result = execute_jump(selected, destination)
                if result is not None:
                    return result

    if "ACTION7" in available and memory.get("initial_undos", 0) < 2:
        memory["initial_undos"] = memory.get("initial_undos", 0)+1
        memory["plan"] = []
        memory["pending"] = None
        return "ACTION7", None, None

    if "RESET" in available:
        memory.clear()
        return "RESET", None, None
    for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        if name in available:
            return name, None, None
    return "ACTION6", 32, 32

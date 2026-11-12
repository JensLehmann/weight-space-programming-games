import math
import time
import heapq
from collections import deque, Counter
import numpy as np


_INF = 10**8
_BASE_EXCLUDED = {0, 1, 3, 4, 5, 7, 9, 12}


def _components(mask):
    """Four-connected components, returned as (left, right, top, bottom, size)."""
    h, w = mask.shape
    a = mask.astype(np.uint8).tolist()
    out = []
    for y in range(h):
        for x in range(w):
            if not a[y][x]:
                continue
            a[y][x] = 0
            stack = [(x, y)]
            l = r = x
            t = b = y
            size = 0
            while stack:
                xx, yy = stack.pop()
                size += 1
                l = min(l, xx)
                r = max(r, xx)
                t = min(t, yy)
                b = max(b, yy)
                for nx, ny in ((xx - 1, yy), (xx + 1, yy),
                               (xx, yy - 1), (xx, yy + 1)):
                    if 0 <= nx < w and 0 <= ny < h and a[ny][nx]:
                        a[ny][nx] = 0
                        stack.append((nx, ny))
            out.append((l, r + 1, t, b + 1, size))
    return out


def _runs(values):
    result = []
    start = None
    for i, value in enumerate(list(values) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            result.append((start, i))
            start = None
    return result


def _clean_frame(obs):
    a = np.asarray(obs, dtype=np.int16).copy()
    # The only interface element overwrites the first display row.
    a[0] = a[1]
    return a


def _orientation(a):
    yellow = a == 4
    coloured = np.ones(a.shape, dtype=bool)
    for colour in _BASE_EXCLUDED:
        coloured &= a != colour

    # Direction from the yellow arrow to its coloured base.
    scores = [
        (int(np.sum(yellow[:-1] & coloured[1:])), 0),       # down
        (int(np.sum(yellow[:, :-1] & coloured[:, 1:])), 3), # right
        (int(np.sum(yellow[:, 1:] & coloured[:, :-1])), 1), # left
        (int(np.sum(yellow[1:] & coloured[:-1])), 2),       # up
    ]
    return max(scores)[1]


def _display_position(k, x, y):
    x, y = int(x), int(y)
    if k == 0:
        return y, x
    if k == 1:
        return x, 63 - y
    if k == 2:
        return 63 - y, 63 - x
    return 63 - x, y


def _characters(board):
    result = []
    for l, r, t, b, size in _components(board == 4):
        if b >= 64 or r - l < 2:
            continue
        vals = [
            int(v) for v in board[b, l:r]
            if int(v) not in _BASE_EXCLUDED
        ]
        if not vals:
            continue
        colour = Counter(vals).most_common(1)[0][0]
        end = b
        while end < 64:
            row = board[end, l:r]
            if np.count_nonzero(row == colour) < max(1, (r - l) * 0.7):
                break
            end += 1
            if end - b >= 8:
                break
        if end > b:
            result.append({
                "l": l, "r": r, "t": t, "b": end,
                "colour": colour, "height": end - t
            })
    return result


def _canonical_characters(state, n, groups):
    if not groups:
        return state
    s = list(state)
    for indices in groups:
        values = sorted(s[n + j] for j in indices)
        for j, value in zip(indices, values):
            s[n + j] = value
    return tuple(s)


def _parse(obs):
    original = _clean_frame(obs)
    k = _orientation(original)
    board = np.rot90(original, k).copy()

    buttons = _components(board == 9)
    if not buttons:
        raise ValueError("No transfer buttons")
    d = Counter(min(r - l, b - t) for l, r, t, b, _ in buttons)
    d = int(d.most_common(1)[0][0])
    if d <= 0:
        raise ValueError("Invalid scale")

    chars = _characters(board)
    if not chars:
        raise ValueError("No visible characters")

    goal_mask = np.ones((64, 64), dtype=bool)
    for colour in _BASE_EXCLUDED:
        goal_mask &= board != colour
    for c in chars:
        goal_mask[c["t"]:c["b"], c["l"]:c["r"]] = False

    goals = []
    for colour in range(16):
        if colour in _BASE_EXCLUDED:
            continue
        for l, r, t, b, count in _components(goal_mask & (board == colour)):
            goals.append((l, r, t, b, colour))

    # Vertical dividers include their gate and target overlays.
    wall_pixels = ((board == 5) | (board == 1) |
                   (board == 12) | goal_mask)
    longest = np.zeros(64, dtype=int)
    current = np.zeros(64, dtype=int)
    for row in wall_pixels:
        current = (current + 1) * row
        longest = np.maximum(longest, current)

    wall_ranges = [
        (l, r) for l, r in _runs(longest >= max(4, 2 * d))
        if r - l <= 2 * d
    ]
    walls = []
    for l, r in wall_ranges:
        ys = np.flatnonzero(np.any(wall_pixels[:, l:r], axis=1))
        if len(ys):
            walls.append((l, r, int(ys[0]), int(ys[-1]) + 1))

    # Horizontal grey pieces are stops between stacked reservoirs.
    floor_mask = board == 5
    floor_mask = floor_mask.copy()
    for l, r, _, _ in walls:
        floor_mask[:, l:r] = False
    floors = []
    for l, r, t, b, count in _components(floor_mask):
        if r - l > 1.5 * d and b - t <= 2 * d:
            floors.append((l, r, t, b))

    # Water rectangles. Buttons may protrude from a rectangle, so use
    # nearly full rows rather than the raw component's vertical bounds.
    water = (board == 0) | (board == 9)
    columns = []
    for l, r, t, b, count in _components(water):
        width = r - l
        if width <= 1.5 * d:
            continue
        section = board[t:b, l:r]
        counts = np.count_nonzero((section == 0) | (section == 9), axis=1)
        zeros = np.count_nonzero(section == 0, axis=1)
        good = np.flatnonzero((counts >= 0.68 * width) & (zeros > 0))
        if not len(good):
            continue
        top = t + int(good[0])
        bottom = t + int(good[-1]) + 1
        columns.append({"l": l, "r": r, "h": top, "base": bottom})

    # A reservoir with zero volume has no black pixels. Its lane, stop,
    # and permanent button still identify it.
    lanes = []
    left = 0
    for l, r, _, _ in sorted(walls):
        if l > left:
            lanes.append((left, l))
        left = max(left, r)
    if left < 64:
        lanes.append((left, 64))

    for l, r in lanes:
        if r - l <= 1.5 * d:
            continue
        bases = {64}
        for fl, fr, ft, fb in floors:
            if fl == l and fr == r:
                bases.add(ft)
        for base in bases:
            if any(p["l"] == l and p["r"] == r and p["base"] == base
                   for p in columns):
                continue
            has_button = any(
                l <= bl and br <= r and bb == base
                for bl, br, bt, bb, _ in buttons
            )
            if has_button:
                columns.append({"l": l, "r": r, "h": base, "base": base})

    columns.sort(key=lambda p: (p["l"], p["base"], p["r"]))
    n = len(columns)
    if not n:
        raise ValueError("No reservoirs")

    def choose_column(candidates, button_top):
        candidates = [
            i for i in candidates if columns[i]["base"] >= button_top
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda i: columns[i]["h"])

    actions = []
    seen_edges = set()
    for bl, br, bt, bb, _ in buttons:
        receiver = choose_column(
            [i for i, p in enumerate(columns)
             if p["l"] <= bl and br <= p["r"]],
            bt
        )
        if receiver is None:
            continue
        p = columns[receiver]
        donor_candidates = []
        if bl == p["l"]:
            for wl, wr, wt, wb in walls:
                if wr == p["l"]:
                    donor_candidates.extend(
                        i for i, q in enumerate(columns) if q["r"] == wl
                    )
        if br == p["r"]:
            for wl, wr, wt, wb in walls:
                if wl == p["r"]:
                    donor_candidates.extend(
                        i for i, q in enumerate(columns) if q["l"] == wr
                    )
        donor = choose_column(donor_candidates, bt)
        if donor is None or donor == receiver:
            continue
        if (donor, receiver) in seen_edges:
            continue
        seen_edges.add((donor, receiver))
        row, col = _display_position(k, (bl + br - 1) // 2,
                                    (bt + bb - 1) // 2)
        actions.append((0, donor, receiver, 0, 0, row, col))

    fills = []
    for p in columns:
        fills.append(max(0, int(round((p["base"] - p["h"]) / d))))

    # Translate the source's movement limits into reservoir capacities.
    positive = k in (0, 3)
    if d % 3 == 0:
        world_step, scale = 3, max(1, d // 3)
    else:
        world_step, scale = 2, max(1, d // 2)
    clearance = (6 if k == 1 and world_step == 3 else 4) * scale

    empty_cap = []
    occupied_cap = []
    adjacency = []
    for i, p in enumerate(columns):
        adjacent = [
            j for j, (wl, wr, wt, wb) in enumerate(walls)
            if wr == p["l"] or wl == p["r"]
        ]
        adjacency.append(adjacent)
        stops = [
            fb for fl, fr, ft, fb in floors
            if fl == p["l"] and fb < p["base"]
        ]
        if stops:
            limit = max(stops) if positive else min(stops)
            occupied_limit = limit - clearance if positive else limit + clearance
        else:
            limit = max((walls[j][2] for j in adjacent), default=0)
            occupied_limit = limit
        empty_cap.append(max(0, int(math.ceil((p["base"] - limit) / d))))
        occupied_cap.append(
            max(0, int(math.ceil((p["base"] - occupied_limit) / d)))
        )

    # Connected transfer components have invariant total volume.
    graph = [[] for _ in columns]
    for kind, i, j, fi, fj, row, col in actions:
        graph[i].append(j)
        graph[j].append(i)
    totals = [0] * n
    visited = set()
    for root in range(n):
        if root in visited:
            continue
        stack = [root]
        visited.add(root)
        group = []
        while stack:
            i = stack.pop()
            group.append(i)
            for j in graph[i]:
                if j not in visited:
                    visited.add(j)
                    stack.append(j)
        total = sum(fills[i] for i in group)
        for i in group:
            totals[i] = total
            empty_cap[i] = min(empty_cap[i], total)
            occupied_cap[i] = min(occupied_cap[i], total)

    max_cap = [
        max(fills[i], empty_cap[i], occupied_cap[i]) for i in range(n)
    ]

    gate_ends = [[] for _ in columns]
    for gl, gr, gt, gb, count in _components((board == 1) | (board == 12)):
        lefts = [i for i, p in enumerate(columns) if p["r"] == gl]
        rights = [i for i, p in enumerate(columns) if p["l"] == gr]
        row, col = _display_position(k, (gl + gr - 1) // 2,
                                    (gt + gb - 1) // 2)
        for i in lefts:
            for j in rights:
                ai = columns[i]["base"] - gb
                aj = columns[j]["base"] - gb
                if ai % d or aj % d:
                    continue
                fi, fj = ai // d, aj // d
                if not (0 <= fi <= max_cap[i] and 0 <= fj <= max_cap[j]):
                    continue
                actions.append((1, i, j, fi, fj, row, col))
                gate_ends[i].append((j, fi, fj))
                gate_ends[j].append((i, fj, fi))

    positions = []
    char_specs = []
    for c in chars:
        possible = [
            i for i, p in enumerate(columns)
            if p["l"] <= c["l"] < p["r"]
        ]
        if not possible:
            raise ValueError("Unattached character")
        owner = min(possible, key=lambda i: abs(columns[i]["h"] - c["b"]))
        if abs(columns[owner]["h"] - c["b"]) > d:
            raise ValueError("Ambiguous character surface")
        positions.append(owner)
        char_specs.append((c["colour"], c["height"]))

    groups_by_spec = {}
    for j, spec in enumerate(char_specs):
        groups_by_spec.setdefault(spec, []).append(j)
    groups = [v for v in groups_by_spec.values() if len(v) > 1]

    target_sets = []
    target_by_col = []
    distance_tables = []
    for colour, char_height in char_specs:
        targets = set()
        for gl, gr, gt, gb, gc in goals:
            if gc != colour:
                continue
            thickness = gb - gt
            target_surface = (gt - 2 * thickness + char_height
                              if positive else gb)
            matching_walls = [
                wi for wi, (wl, wr, wt, wb) in enumerate(walls)
                if max(gl, wl) < min(gr, wr)
            ]
            for i, p in enumerate(columns):
                if not any(wi in adjacency[i] for wi in matching_walls):
                    continue
                amount = p["base"] - target_surface
                if amount % d:
                    continue
                f = amount // d
                if 0 <= f <= max_cap[i]:
                    targets.add((i, f))
        target_sets.append(targets)

        by_col = [[] for _ in columns]
        for i, f in targets:
            by_col[i].append(f)
        target_by_col.append(by_col)

        # Relaxed single-character problem: height changes cost one,
        # and crossing an aligned gate costs one.
        dist = [[_INF] * (max_cap[i] + 1) for i in range(n)]
        queue = deque()
        for i, f in targets:
            dist[i][f] = 0
            queue.append((i, f))
        while queue:
            i, f = queue.popleft()
            nd = dist[i][f] + 1
            if f > 0 and nd < dist[i][f - 1]:
                dist[i][f - 1] = nd
                queue.append((i, f - 1))
            if f < max_cap[i] and nd < dist[i][f + 1]:
                dist[i][f + 1] = nd
                queue.append((i, f + 1))
            for j, fi, fj in gate_ends[i]:
                if f == fi and nd < dist[j][fj]:
                    dist[j][fj] = nd
                    queue.append((j, fj))
        distance_tables.append(dist)

    state = tuple(fills + positions)
    state = _canonical_characters(state, n, groups)
    model = {
        "k": k, "d": d, "columns": columns, "n": n,
        "m": len(chars), "actions": actions,
        "empty": empty_cap, "occupied": occupied_cap,
        "max_cap": max_cap, "gates": gate_ends,
        "targets": target_sets, "target_by_col": target_by_col,
        "dist": distance_tables, "groups": groups,
        "one_per_column": len(set(positions)) == len(positions)
    }
    return model, state, board


def _won(model, state):
    n = model["n"]
    return all(
        (state[n + j], state[state[n + j]]) in model["targets"][j]
        for j in range(model["m"])
    )


def _heuristic(model, state):
    n = model["n"]
    largest = 0
    total_relaxed = 0
    for c in range(model["m"]):
        i = state[n + c]
        f = state[i]
        table = model["dist"][c]
        if f >= len(table[i]):
            return _INF
        relaxed = table[i][f]
        if relaxed >= _INF:
            return _INF
        total_relaxed += relaxed

        best = _INF
        for target in model["target_by_col"][c][i]:
            best = min(best, abs(f - target))

        # Before the first crossing, both participating surfaces must
        # reach the gate. A transfer cannot raise both simultaneously.
        for j, fi, fj in model["gates"][i]:
            a = fi - f
            b = fj - state[j]
            alignment = max(abs(a), abs(b), abs(a + b))
            after = table[j][fj]
            best = min(best, alignment + 1 + after)
        largest = max(largest, best)

    if model["one_per_column"]:
        largest = max(largest, (total_relaxed + 1) // 2)
    return largest


def _successor(model, state, aid):
    kind, i, j, fi, fj, row, col = model["actions"][aid]
    n = model["n"]
    s = list(state)
    if kind == 0:
        occupied = j in state[n:]
        cap = model["occupied"][j] if occupied else model["empty"][j]
        if state[i] <= 0 or state[j] >= cap:
            return None
        s[i] -= 1
        s[j] += 1
    else:
        if state[i] != fi or state[j] != fj:
            return None
        moved = False
        for c in range(model["m"]):
            pos = s[n + c]
            if pos == i:
                s[n + c] = j
                moved = True
            elif pos == j:
                s[n + c] = i
                moved = True
        if not moved:
            return None
    result = _canonical_characters(tuple(s), n, model["groups"])
    return None if result == state else result


def _search(model, start, remaining, banned, visits):
    if _won(model, start):
        return []
    initial_h = _heuristic(model, start)
    if initial_h >= _INF:
        return None

    deadline = time.perf_counter() + 6.5
    frontier = []
    serial = 0
    weight = 1.55
    heapq.heappush(frontier, (weight * initial_h, initial_h, 0, serial, start))
    best_g = {start: 0}
    parent = {}
    best_partial = start
    best_partial_rank = (initial_h, 0)
    solution = None
    expansions = 0

    while frontier:
        _, h, g, _, state = heapq.heappop(frontier)
        if best_g.get(state) != g:
            continue
        if _won(model, state):
            solution = state
            break
        if g >= remaining:
            continue

        expansions += 1
        if expansions % 256 == 0:
            if time.perf_counter() >= deadline or expansions >= 220000:
                break

        for aid in range(len(model["actions"])):
            if (state, aid) in banned:
                continue
            nxt = _successor(model, state, aid)
            if nxt is None:
                continue
            ng = g + 1
            if ng >= best_g.get(nxt, _INF):
                continue
            nh = _heuristic(model, nxt)
            if nh >= _INF or ng + nh > remaining:
                continue
            best_g[nxt] = ng
            parent[nxt] = (state, aid)
            serial += 1
            heapq.heappush(
                frontier, (ng + weight * nh, nh, ng, serial, nxt)
            )
            rank = (nh + 2 * visits.get(nxt, 0), ng)
            if rank < best_partial_rank:
                best_partial_rank = rank
                best_partial = nxt

    endpoint = solution if solution is not None else best_partial
    if endpoint == start:
        return None

    path = []
    while endpoint != start:
        previous, aid = parent[endpoint]
        path.append(aid)
        endpoint = previous
    path.reverse()
    return path


def _surface_consistent(model, state, board):
    d = model["d"]
    for i, p in enumerate(model["columns"]):
        h = p["base"] - d * state[i]
        x = (p["l"] + p["r"] - 1) // 2
        if state[i] == 0:
            y = p["base"] - 1
            if 0 <= y < 64 and board[y, x] == 0:
                return False
        else:
            y = max(0, h)
            if y >= 64 or board[y, x] not in (0, 9):
                return False
            if 0 < h < 64 and board[h - 1, x] == 0:
                return False
    return True


def _fallback(obs, memory):
    a = np.asarray(obs)
    boxes = _components(a == 9)
    if boxes:
        counter = memory.get("_vc_fallback", 0)
        memory["_vc_fallback"] = counter + 1
        # Change direction periodically rather than repeatedly hitting
        # the same saturated reservoir.
        index = (counter // 3) % len(boxes)
        l, r, t, b, _ = boxes[index]
        return "ACTION6", (t + b - 1) // 2, (l + r - 1) // 2
    return "ACTION6", 1, 1


def act(obs, available, memory):
    """Return (action name, display row, display column)."""
    try:
        cache = memory.get("_vc")
        rebuilt = False

        if cache is not None:
            model = cache["model"]
            board = np.rot90(_clean_frame(obs), model["k"]).copy()
            state = cache["expected"]

            # A rejected move changes only the progress bar.
            if (cache.get("previous_board") is not None and
                    np.array_equal(board, cache["previous_board"])):
                previous = cache.get("previous_state")
                aid = cache.get("last_action")
                if previous is not None and aid is not None and state != previous:
                    cache["banned"].add((previous, aid))
                    state = previous
                    cache["expected"] = previous
                    cache["plan"] = []

            if not _surface_consistent(model, state, board):
                cache = None

        if cache is None:
            model, state, board = _parse(obs)
            cache = {
                "model": model,
                "expected": state,
                "plan": [],
                "banned": set(),
                "visits": {},
                "previous_board": None,
                "previous_state": None,
                "last_action": None,
            }
            memory["_vc"] = cache
            rebuilt = True

        cache["visits"][state] = cache["visits"].get(state, 0) + 1

        if _won(model, state):
            # A harmless click also resolves an already-satisfied state.
            background = np.argwhere(np.asarray(obs)[1:] == 3)
            if len(background):
                y, x = background[0]
                return "ACTION6", int(y) + 1, int(x)
            return "ACTION6", 1, 1

        remaining = max(
            1, min(50, int(round(np.count_nonzero(np.asarray(obs)[0] == 7)
                                * 50 / 64)))
        )

        plan = cache["plan"]
        if plan:
            candidate = _successor(model, state, plan[0])
            if candidate is None or (state, plan[0]) in cache["banned"]:
                plan = []
                cache["plan"] = []

        if not plan:
            plan = _search(
                model, state, remaining,
                cache["banned"], cache["visits"]
            )
            cache["plan"] = plan or []

        if cache["plan"]:
            aid = cache["plan"].pop(0)
            nxt = _successor(model, state, aid)
        else:
            # Receding-horizon escape if the complete search was cut off.
            choices = []
            for candidate_aid in range(len(model["actions"])):
                if (state, candidate_aid) in cache["banned"]:
                    continue
                candidate = _successor(model, state, candidate_aid)
                if candidate is None:
                    continue
                score = (_heuristic(model, candidate)
                         + 3 * cache["visits"].get(candidate, 0))
                choices.append((score, candidate_aid, candidate))
            if not choices:
                memory.pop("_vc", None)
                return _fallback(obs, memory)
            _, aid, nxt = min(choices)

        if nxt is None:
            memory.pop("_vc", None)
            return _fallback(obs, memory)

        cache["previous_board"] = board.copy()
        cache["previous_state"] = state
        cache["last_action"] = aid
        cache["expected"] = nxt

        action = model["actions"][aid]
        return "ACTION6", int(action[5]), int(action[6])

    except Exception:
        memory.pop("_vc", None)
        return _fallback(obs, memory)

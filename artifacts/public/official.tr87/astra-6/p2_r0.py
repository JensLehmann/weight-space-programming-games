import heapq
import itertools
import time
import numpy as np


# Each entry describes the seven symbols in one alphabet, in cycle order.
_SHAPES = {
    10: [
        "00100/00100/01110/00100/11111",
        "00001/00101/11111/10100/10000",
        "11111/10100/10000/10100/11111",
        "00100/11111/10101/10101/00100",
        "10001/11111/00100/11111/10001",
        "11111/00100/11100/10000/11111",
        "10001/11111/10001/10001/11011",
    ],
    7: [
        "10000/11110/10010/10010/11111",
        "11111/10001/11101/10101/11111",
        "00111/00101/11111/10100/11100",
        "01110/01010/11111/10001/11111",
        "11111/10001/10001/11111/00100",
        "11110/10011/10001/11001/01111",
        "00100/11111/10101/11111/00100",
    ],
    11: [
        "10101/10000/10101/10000/11111",
        "11101/10000/00000/00001/10111",
        "10111/00100/00100/00100/11101",
        "00100/10101/00100/10101/00100",
        "11011/10001/10101/10001/11011",
        "10111/10101/11101/00000/10101",
        "10001/00000/10001/11011/01010",
    ],
}


def _canonical(a):
    a = np.asarray(a, dtype=np.uint8)
    keys = []
    for _ in range(4):
        key = 0
        for i, v in enumerate(a.flat):
            key |= int(v) << i
        keys.append(key)
        a = np.rot90(a)
    return min(keys)


_BASE = {}
for _color, _entries in _SHAPES.items():
    _BASE[_color] = {}
    for _index, _entry in enumerate(_entries):
        _a = [[int(c) for c in row] for row in _entry.split("/")]
        _BASE[_color][_canonical(_a)] = _index


def _read_board(obs):
    """Find the rectangular tile strips, then split them into seven-pixel tiles."""
    a = np.asarray(obs)
    grid = a.tolist()
    seen = np.zeros((64, 64), dtype=np.uint8)
    tiles = []

    for y in range(63):
        for x in range(64):
            color = grid[y][x]
            if seen[y, x] or color in (0, 2, 3, 5):
                continue
            seen[y, x] = 1
            stack = [(x, y)]
            xmin = xmax = x
            ymin = ymax = y
            while stack:
                xx, yy = stack.pop()
                xmin = min(xmin, xx)
                xmax = max(xmax, xx)
                ymin = min(ymin, yy)
                ymax = max(ymax, yy)
                for nx, ny in ((xx-1, yy), (xx+1, yy),
                               (xx, yy-1), (xx, yy+1)):
                    if (0 <= nx < 64 and 0 <= ny < 63
                            and not seen[ny, nx]
                            and grid[ny][nx] == color):
                        seen[ny, nx] = 1
                        stack.append((nx, ny))

            width = xmax - xmin + 1
            if ymax - ymin + 1 != 7 or width % 7:
                continue
            if not (np.all(a[ymin, xmin:xmax+1] == color)
                    and np.all(a[ymax, xmin:xmax+1] == color)):
                continue

            for tx in range(xmin, xmax + 1, 7):
                patch = a[ymin:ymin+7, tx:tx+7]
                if (patch.shape != (7, 7)
                        or not np.all(patch[:, 0] == color)
                        or not np.all(patch[:, -1] == color)):
                    continue
                ink = a[ymin+1:ymin+6, tx+1:tx+6] == 5
                tiles.append({
                    "x": tx, "y": ymin, "c": int(color),
                    "k": _canonical(ink)
                })

    tiles.sort(key=lambda t: (t["y"], t["x"]))
    if not tiles:
        return None

    split = None
    for y in range(15, 60):
        if np.count_nonzero(a[y] == 3) >= 56:
            split = y
            break

    ys = sorted(set(t["y"] for t in tiles))
    if split is None:
        if len(ys) < 3:
            return None
        split = ys[-2] - 1

    bottom = [t for t in tiles if t["y"] > split]
    by = sorted(set(t["y"] for t in bottom))
    if len(by) < 2:
        return None
    source = sorted((t for t in bottom if t["y"] == by[0]),
                    key=lambda t: t["x"])
    target = sorted((t for t in bottom if t["y"] == by[1]),
                    key=lambda t: t["x"])

    rules = []
    top = [t for t in tiles if t["y"] < split]
    for y in sorted(set(t["y"] for t in top)):
        row = sorted((t for t in top if t["y"] == y),
                     key=lambda t: t["x"])
        groups = []
        for t in row:
            if groups and t["x"] == groups[-1][-1]["x"] + 7:
                groups[-1].append(t)
            else:
                groups.append([t])
        for left, right in zip(groups, groups[1:]):
            lo = left[-1]["x"] + 7
            hi = right[0]["x"]
            if 0 < hi - lo <= 6 and np.all(a[y+3, lo:hi] == 3):
                rules.append((left, right))

    if not rules:
        return None

    top_controls = [g for pair in rules for g in pair]
    bottom_controls = [[t] for t in target]

    def marker_score(group):
        t = group[0]
        x, y = t["x"] + 1, t["y"]
        w = 5 + 7 * (len(group) - 1)
        score = 0
        if x + w <= 64:
            if y >= 3 and np.all(a[y-3, x:x+w] == 0):
                score += 2
            if y + 9 < 63 and np.all(a[y+9, x:x+w] == 0):
                score += 2
        return score

    ts = [marker_score(g) for g in top_controls]
    bs = [marker_score(g) for g in bottom_controls]
    alter = max(ts, default=0) > max(bs, default=0)
    controls = top_controls if alter else bottom_controls
    scores = ts if alter else bs
    selected = max(range(len(scores)), key=lambda i: scores[i])

    return tiles, source, target, rules, controls, selected, alter


def _navigation(start, finish, n):
    clockwise = (finish - start) % n
    counterclockwise = (start - finish) % n
    return "ACTION4" if clockwise <= counterclockwise else "ACTION3"


def _make_plan(shifts, start):
    """Shortest covering walk on a cycle; edits are performed on first visit."""
    n = len(shifts)
    shifts = [v % 7 for v in shifts]
    needed = {i for i, v in enumerate(shifts) if v}
    if not needed:
        # A symbol edit, rather than cursor movement, triggers the goal check.
        return ["ACTION2", "ACTION1"]

    best_path = None
    for direction in (1, -1):
        for first_leg in range(n):
            p = start
            path = [p]
            remaining = needed - {p}
            if remaining:
                directions = ([direction] * first_leg
                              + [-direction] * (n - 1))
                for d in directions:
                    p = (p + d) % n
                    path.append(p)
                    remaining.discard(p)
                    if not remaining:
                        break
            if not remaining and (best_path is None
                                  or len(path) < len(best_path)):
                best_path = path

    if best_path is None:
        best_path = [(start + k) % n for k in range(n)]

    result = []
    edited = set()
    previous = None
    for p in best_path:
        if previous is not None:
            result.append("ACTION4" if (p - previous) % n == 1
                          else "ACTION3")
        if p not in edited and shifts[p]:
            d = shifts[p]
            result.extend(["ACTION2"] * d if d <= 3
                          else ["ACTION1"] * (7 - d))
            edited.add(p)
        previous = p
    return result


def _solve(source, target, rules, controls, selected, alter,
           failed, tried, deadline):
    """
    Match symbolic productions using modular equality constraints.
    Every controllable group has one shift variable modulo seven.
    """
    n = len(controls)
    constant = n

    def term(t, variable):
        return (t["c"], t["v"], variable)

    S = [term(t, constant) for t in source]
    T = [term(t, constant if alter else i)
         for i, t in enumerate(target)]
    R = []
    for i, (left, right) in enumerate(rules):
        lv = 2*i if alter else constant
        rv = 2*i+1 if alter else constant
        R.append(([term(t, lv) for t in left],
                  [term(t, rv) for t in right]))

    initial = (list(range(n+1)), [0] * (n+1))
    answers = {}
    seen_leaves = set()
    seen_assignments = set()
    work = [0]
    local_work = [0]
    current_mode = [None]

    def find(state, v):
        parent, weight = state
        offset = 0
        while parent[v] != v:
            offset += weight[v]
            v = parent[v]
        return v, offset % 7

    def allowed():
        work[0] += 1
        local_work[0] += 1
        if local_work[0] > 26000:
            return False
        if (work[0] & 127) == 0 and time.monotonic() >= deadline:
            return False
        return True

    def unify(state, aa, bb):
        if len(aa) != len(bb):
            return None
        for a, b in zip(aa, bb):
            if a[0] != b[0]:
                return None
        if not allowed():
            return None
        p, w = state[0][:], state[1][:]
        out = (p, w)
        for a, b in zip(aa, bb):
            ra, oa = find(out, a[2])
            rb, ob = find(out, b[2])
            difference = (b[1] - a[1] - oa + ob) % 7
            if ra == rb:
                if difference:
                    return None
            else:
                p[ra] = rb
                w[ra] = difference
        return out

    def match_part(state, sequence, pos, pattern):
        if pos + len(pattern) > len(sequence):
            return None
        return unify(state, sequence[pos:pos+len(pattern)], pattern)

    def concrete(terms, shifts):
        return [c * 7 + ((v + (shifts[q] if q < n else 0)) % 7)
                for c, v, q in terms]

    def check(shifts, mode):
        ss = concrete(S, shifts)
        tt = concrete(T, shifts)
        rr = [(concrete(l, shifts), concrete(r, shifts)) for l, r in R]

        def translate(seq, kind):
            result = []
            pos = 0
            while pos < len(seq):
                found = False
                for left, right in rr:
                    if seq[pos:pos+len(left)] != left:
                        continue
                    output = right
                    if kind == "tree":
                        output = []
                        valid = True
                        for symbol in right:
                            for l2, r2 in rr:
                                if l2[0] == symbol:
                                    output.extend(r2)
                                    break
                            else:
                                valid = False
                                break
                        if not valid:
                            continue
                    elif kind == "double":
                        for l2, r2 in rr:
                            if all(a == b for a, b in zip(right, l2)):
                                output = r2
                                break
                        else:
                            continue
                    result.extend(output)
                    pos += len(left)
                    found = True
                    break
                if not found or len(result) > 80:
                    return None
            return result

        if mode == "stream":
            intermediate = translate(ss, "direct")
            out = None if intermediate is None else translate(
                intermediate, "direct")
        else:
            out = translate(ss, mode)
        if out is None or len(out) > len(tt) or tt[:len(out)] != out:
            return None
        return len(out)

    def leaf(state, mode):
        if time.monotonic() >= deadline:
            return

        roots = [find(state, i) for i in range(n+1)]
        members = {}
        for i, (r, offset) in enumerate(roots):
            members.setdefault(r, []).append((i, offset))
        representative = {r: min(i for i, _ in g)
                          for r, g in members.items()}
        anchor_offsets = {
            r: next(o for i, o in g if i == representative[r])
            for r, g in members.items()
        }
        normalized = tuple((representative[r],
                            (o - anchor_offsets[r]) % 7)
                           for r, o in roots)
        key = (mode, normalized)
        if key in seen_leaves:
            return
        seen_leaves.add(key)

        constant_root, constant_offset = roots[n]
        options = []
        for r, group in members.items():
            variable_members = [(i, o) for i, o in group if i < n]
            if not variable_members:
                continue
            values = [(-constant_offset) % 7] if r == constant_root else range(7)
            choices = []
            for root_value in values:
                assignment = [(i, (root_value + o) % 7)
                              for i, o in variable_members]
                cost = sum(min(v, 7-v) for _, v in assignment)
                changed = sum(v != 0 for _, v in assignment)
                choices.append((cost, changed, assignment))
            choices.sort(key=lambda z: (z[0], z[1]))
            options.append(choices)

        if not options:
            return
        start_indices = (0,) * len(options)
        minimum = sum(g[0][0] for g in options)
        old = answers.get(mode)
        if old is not None and minimum > old[0]:
            return

        heap = [(minimum, start_indices)]
        visited = {start_indices}
        # Free, unused rules occasionally need a small edit to avoid shadowing
        # a later production. Try low-cost alternatives for those variables.
        for attempt in range(40):
            if not heap or time.monotonic() >= deadline:
                break
            cost, indices = heapq.heappop(heap)
            old = answers.get(mode)
            if old is not None and cost > old[0]:
                break
            shifts = [0] * n
            for choices, index in zip(options, indices):
                for i, value in choices[index][2]:
                    shifts[i] = value

            assignment_key = (mode, tuple(shifts))
            if assignment_key not in seen_assignments:
                seen_assignments.add(assignment_key)
                consumed = check(shifts, mode)
                if consumed is not None:
                    signature = tuple(
                        (t["c"], (t["v"] + shifts[i]) % 7)
                        for i, group in enumerate(controls) for t in group)
                    if signature not in tried:
                        plan = _make_plan(shifts, selected)
                        # Full-length translations are a useful discriminator
                        # between superficially plausible multi-stage models.
                        penalty = 8 * (len(target) - consumed)
                        prior = {"direct": 0.0, "tree": 0.2,
                                 "stream": 0.4, "double": 0.8}[mode]
                        score = len(plan) + penalty + prior
                        old = answers.get(mode)
                        if old is None or score < old[0]:
                            answers[mode] = (score, plan, signature)
                        # The cheapest free-variable realization is sufficient
                        # once it passes the deterministic rule-order check.
                        break

            for j in range(len(options)):
                if indices[j] + 1 >= len(options[j]):
                    continue
                nxt = list(indices)
                nxt[j] += 1
                nxt = tuple(nxt)
                if nxt not in visited:
                    visited.add(nxt)
                    new_cost = (cost - options[j][indices[j]][0]
                                + options[j][nxt[j]][0])
                    heapq.heappush(heap, (new_cost, nxt))

    def direct_search(si, ti, state, mode):
        if local_work[0] > 26000 or time.monotonic() >= deadline:
            return
        if si >= len(S):
            leaf(state, mode)
            return

        for left, right in R:
            st = match_part(state, S, si, left)
            if st is None:
                continue
            next_si = si + len(left)
            if mode == "direct":
                st2 = match_part(st, T, ti, right)
                if st2 is not None:
                    direct_search(next_si, ti+len(right), st2, mode)

            elif mode == "double":
                for l2, r2 in R:
                    length = min(len(right), len(l2))
                    st2 = unify(st, right[:length], l2[:length])
                    if st2 is not None:
                        st3 = match_part(st2, T, ti, r2)
                        if st3 is not None:
                            direct_search(next_si, ti+len(r2), st3, mode)

            elif mode == "tree":
                def expand(k, target_pos, tree_state):
                    if local_work[0] > 26000:
                        return
                    if k == len(right):
                        direct_search(next_si, target_pos, tree_state, mode)
                        return
                    symbol = right[k]
                    for l2, r2 in R:
                        st2 = unify(tree_state, [symbol], [l2[0]])
                        if st2 is not None:
                            st3 = match_part(st2, T, target_pos, r2)
                            if st3 is not None:
                                expand(k+1, target_pos+len(r2), st3)
                expand(0, ti, st)

    def stream_second(intermediate, pos, ti, state):
        if local_work[0] > 26000 or time.monotonic() >= deadline:
            return
        if pos == len(intermediate):
            leaf(state, "stream")
            return
        for left, right in R:
            st = match_part(state, intermediate, pos, left)
            if st is not None:
                st2 = match_part(st, T, ti, right)
                if st2 is not None:
                    stream_second(intermediate, pos+len(left),
                                  ti+len(right), st2)

    second_families = {left[0][0] for left, _ in R}

    def stream_first(si, intermediate, state):
        if local_work[0] > 26000 or time.monotonic() >= deadline:
            return
        if si == len(S):
            stream_second(intermediate, 0, 0, state)
            return
        for left, right in R:
            if any(t[0] not in second_families for t in right):
                continue
            st = match_part(state, S, si, left)
            if st is not None:
                stream_first(si+len(left), intermediate+right, st)

    for mode in ("direct", "tree", "stream", "double"):
        if time.monotonic() >= deadline:
            break
        limit = 4 if mode == "stream" else 1
        if failed.get(mode, 0) >= limit:
            continue
        current_mode[0] = mode
        local_work[0] = 0
        if mode == "stream":
            stream_first(0, [], initial)
        else:
            direct_search(0, 0, initial, mode)

    if not answers:
        return None
    mode, answer = min(answers.items(), key=lambda item: item[1][0])
    return mode, answer[1], answer[2]


def act(obs, available, memory):
    def emit(action):
        if action not in available:
            action = next((a for a in available if a != "RESET"),
                          available[0] if available else "ACTION1")
        return action, None, None

    queue = memory.get("_queue")
    if queue:
        return emit(queue.pop(0))

    # If another observation arrives after a supposedly completing edit, that
    # interpretation did not describe this level. Try another interpretation.
    previous = memory.pop("_attempt", None)
    if previous is not None:
        mode, signature = previous
        failed = memory.setdefault("_failed", {})
        failed[mode] = failed.get(mode, 0) + 1
        memory.setdefault("_tried", set()).add(signature)

    board = _read_board(obs)
    if board is None:
        return emit("ACTION4")

    tiles, source, target, rules, controls, selected, alter = board
    maps = memory.setdefault("_maps", {})
    trained = memory.setdefault("_trained", set())

    for color, mapping in _BASE.items():
        if color not in maps:
            maps[color] = dict(mapping)

    mutable_colors = {t["c"] for g in controls for t in g}
    unknown = []
    for color in mutable_colors:
        observed_keys = {t["k"] for t in tiles if t["c"] == color}
        if color not in trained:
            if color not in _BASE or not observed_keys.issubset(_BASE[color]):
                unknown.append(color)

    # Unseen alphabets can be calibrated by cycling one controllable symbol.
    # Six increments expose the complete seven-symbol cycle.
    learning = memory.get("_learning")
    if learning is None and unknown:
        color = min(unknown)
        index = next(i for i, group in enumerate(controls)
                     if any(t["c"] == color for t in group))
        learning = {"color": color, "index": index, "count": -1}
        memory["_learning"] = learning

    if learning is not None:
        color = learning["color"]
        index = learning["index"]
        if selected != index:
            return emit(_navigation(selected, index, len(controls)))
        tile = next(t for t in controls[index] if t["c"] == color)
        count = learning["count"]
        if count < 0:
            maps[color] = {tile["k"]: 0}
            learning["count"] = 1
            return emit("ACTION2")
        maps[color][tile["k"]] = count % 7
        if count < 6:
            learning["count"] = count + 1
            return emit("ACTION2")
        trained.add(color)
        del memory["_learning"]
        # Other unknown controllable alphabets are handled on the next call.
        if any(c != color for c in unknown):
            return emit("ACTION4")

    # A fixed alphabet needs only consistent identities, not its cycle order.
    for color in {t["c"] for t in tiles}:
        mapping = maps.setdefault(color, {})
        missing = sorted({t["k"] for t in tiles
                          if t["c"] == color and t["k"] not in mapping})
        unused = [i for i in range(7) if i not in mapping.values()]
        for key, value in zip(missing, unused):
            mapping[key] = value

    for tile in tiles:
        mapping = maps[tile["c"]]
        if tile["k"] not in mapping:
            return emit("ACTION2")
        tile["v"] = mapping[tile["k"]]

    deadline = time.monotonic() + 7.5
    solution = _solve(
        source, target, rules, controls, selected, alter,
        memory.setdefault("_failed", {}),
        memory.setdefault("_tried", set()),
        deadline
    )

    if solution is not None:
        mode, plan, signature = solution
        memory["_attempt"] = (mode, signature)
        memory["_queue"] = plan[1:]
        return emit(plan[0])

    # Retry from the clean puzzle once if recognition/search was insufficient.
    if not memory.get("_reset_used") and "RESET" in available:
        memory.clear()
        memory["_reset_used"] = True
        return emit("RESET")

    # Keep exploring rather than repeatedly issuing an invalid or inert action.
    tick = memory.get("_fallback", 0)
    memory["_fallback"] = tick + 1
    if tick % 7 == 6:
        return emit("ACTION4")
    return emit("ACTION2")

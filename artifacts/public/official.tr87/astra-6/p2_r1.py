import time
import heapq
import numpy as np


_SHAPES = {
    10: [
        ("00100", "00100", "01110", "00100", "11111"),
        ("00001", "00101", "11111", "10100", "10000"),
        ("11111", "10100", "10000", "10100", "11111"),
        ("00100", "11111", "10101", "10101", "00100"),
        ("10001", "11111", "00100", "11111", "10001"),
        ("11111", "00100", "11100", "10000", "11111"),
        ("10001", "11111", "10001", "10001", "11011"),
    ],
    7: [
        ("10000", "11110", "10010", "10010", "11111"),
        ("11111", "10001", "11101", "10101", "11111"),
        ("00111", "00101", "11111", "10100", "11100"),
        ("01110", "01010", "11111", "10001", "11111"),
        ("11111", "10001", "10001", "11111", "00100"),
        ("11110", "10011", "10001", "11001", "01111"),
        ("00100", "11111", "10101", "11111", "00100"),
    ],
    11: [
        ("10101", "10000", "10101", "10000", "11111"),
        ("11101", "10000", "00000", "00001", "10111"),
        ("10111", "00100", "00100", "00100", "11101"),
        ("00100", "10101", "00100", "10101", "00100"),
        ("11011", "10001", "10101", "10001", "11011"),
        ("10111", "10101", "11101", "00000", "10101"),
        ("10001", "00000", "10001", "11011", "01010"),
    ],
}


def _canonical(rows):
    a = [list(row) for row in rows]
    values = []
    for _ in range(4):
        bits = 0
        for row in a:
            for b in row:
                bits = (bits << 1) | int(b)
        values.append(bits)
        a = [list(row) for row in zip(*a[::-1])]
    return min(values)


def _initial_maps():
    result = {}
    for color, shapes in _SHAPES.items():
        result[color] = {
            _canonical([[int(b) for b in row] for row in shape]): i
            for i, shape in enumerate(shapes)
        }
    return result


def _token(cell):
    return (cell["c"], cell["m"])


def _parse(obs, old_mode=False):
    cells = []
    occupied = np.zeros((64, 64), dtype=bool)

    for y in range(58):
        x = 0
        while x < 64:
            color = int(obs[y, x])
            end = x + 1
            while end < 64 and int(obs[y, end]) == color:
                end += 1
            if color not in (2, 3, 5) and end - x >= 7:
                for xx in range(x, end - 6, 7):
                    if occupied[y:y + 7, xx:xx + 7].any():
                        continue
                    a = obs[y:y + 7, xx:xx + 7]
                    if not (
                        np.all(a[0] == color)
                        and np.all(a[6] == color)
                        and np.all(a[:, 0] == color)
                        and np.all(a[:, 6] == color)
                    ):
                        continue
                    middle = a[1:6, 1:6]
                    if not np.all((middle == color) | (middle == 5)):
                        continue
                    if not np.any(middle == 5):
                        continue
                    cells.append({
                        "x": xx,
                        "y": y,
                        "c": color,
                        "m": _canonical((middle == 5).astype(int).tolist()),
                    })
                    occupied[y:y + 7, xx:xx + 7] = True
            x = end

    if len(cells) < 4:
        return None

    sep = None
    for y in range(18, 61):
        if np.count_nonzero(obs[y] == 3) >= 61:
            sep = y
            break

    ys = sorted(set(c["y"] for c in cells))
    if sep is None:
        if len(ys) < 3:
            return None
        sep = ys[-2]

    top = sorted(
        (c for c in cells if c["y"] < sep),
        key=lambda c: (c["y"], c["x"]),
    )
    bottom = sorted(
        (c for c in cells if c["y"] >= sep),
        key=lambda c: (c["y"], c["x"]),
    )
    if not top or not bottom:
        return None

    iy = bottom[0]["y"]
    inp = [c for c in bottom if c["y"] == iy]
    out = [c for c in bottom if c["y"] != iy]
    if not out:
        return None

    by_y = {}
    for c in top:
        by_y.setdefault(c["y"], []).append(c)

    rules = []
    for y, row in sorted(by_y.items()):
        groups = []
        for c in row:
            if groups and c["x"] == groups[-1][-1]["x"] + 7:
                groups[-1].append(c)
            else:
                groups.append([c])
        for left, right in zip(groups, groups[1:]):
            lo = left[-1]["x"] + 7
            hi = right[0]["x"]
            if hi > lo and np.any(obs[y + 3, lo:hi] == 3):
                rules.append((left, right))

    if not rules:
        return None

    rule_groups = [g for rule in rules for g in rule]

    def bracket(group):
        c = group[0]
        yy, xx = c["y"] - 3, c["x"] + 1
        width = 7 * (len(group) - 1) + 5
        return (
            yy >= 0
            and xx + width <= 64
            and np.all(obs[yy, xx:xx + width] == 0)
        )

    selected_top = next(
        (i for i, g in enumerate(rule_groups) if bracket(g)), None
    )
    selected_bottom = next(
        (i for i, c in enumerate(out) if bracket([c])), None
    )

    if selected_top is not None:
        alter, selected = True, selected_top
    elif selected_bottom is not None:
        alter, selected = False, selected_bottom
    else:
        alter, selected = bool(old_mode), 0

    editable = rule_groups if alter else [[c] for c in out]
    key = (
        tuple(
            tuple((c["x"], c["y"], c["c"]) for c in g)
            for g in rule_groups
        ),
        tuple((c["x"], c["y"], c["c"]) for c in inp + out),
        alter,
    )
    return {
        "rules": rules,
        "inp": inp,
        "out": out,
        "editable": editable,
        "selected": selected,
        "alter": alter,
        "key": key,
    }


def _route(n, selected, dirty):
    if not dirty:
        return 0, 1
    distances = sorted(set((i - selected) % n for i in dirty))
    best = (10**9, 1)
    for right in [0] + distances:
        left = max((n - d for d in distances if d > right), default=0)

        cost = 2 * right + left
        direction = 1 if right else -1
        if cost < best[0]:
            best = (cost, direction)

        cost = right + 2 * left
        direction = -1 if left else 1
        if cost < best[0]:
            best = (cost, direction)
    return best


def _target_cost(current, target, selected, maps):
    dirty = []
    edits = 0
    for i, (a, b) in enumerate(zip(current, target)):
        if a == b:
            continue
        if a[0] != b[0]:
            return float("inf")
        dirty.append(i)
        lookup = maps.get(a[0], {})
        if a[1] in lookup and b[1] in lookup:
            d = (lookup[b[1]] - lookup[a[1]]) % 7
            edits += min(d, 7 - d)
        else:
            edits += 3
    if not dirty:
        return 2
    return edits + _route(len(current), selected, dirty)[0]


def _configurations(rules, cap=1000):
    n = len(rules)
    candidates = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            li, ri = rules[i]
            lj, rj = rules[j]
            if li[0][0] == lj[0][0] and ri[0][0] == rj[0][0]:
                candidates.append((i, j))

    result = [()]
    frontier = [((), frozenset(), -1)]
    for _ in range(n // 2):
        following = []
        for pairs, used, last in frontier:
            for k in range(last + 1, len(candidates)):
                i, j = candidates[k]
                if i in used or j in used:
                    continue
                new_pairs = pairs + ((i, j),)
                result.append(new_pairs)
                following.append((new_pairs, used | {i, j}, k))
                if len(result) >= cap:
                    return result
        frontier = following
        if not frontier:
            break
    return result


def _effective(rules, pairs):
    links = dict(pairs)
    skipped = {j for _, j in pairs}
    result = []
    for i, (left, right) in enumerate(rules):
        if i in skipped:
            continue
        if i in links:
            l2, r2 = rules[links[i]]
            result.append((left, left + l2, right + r2))
        else:
            result.append((left, left, right))
    return result


def _translate(inp, rules, mode, pairs, limit):
    effective = _effective(rules, pairs) if mode == "double" else None
    pos = 0
    result = []
    penalty = 0.0

    while pos < len(inp):
        found = False
        if mode == "double":
            for original_left, left, middle in effective:
                if tuple(inp[pos:pos + len(original_left)]) != tuple(original_left):
                    continue
                for _, second_left, right in effective:
                    if all(a == b for a, b in zip(middle, second_left)):
                        if len(middle) != len(second_left):
                            penalty += 5
                        if tuple(inp[pos:pos + len(left)]) != tuple(left):
                            penalty += 3
                        result.extend(right)
                        pos += len(left)
                        found = True
                        break
                if found:
                    break
        else:
            for left, right in rules:
                if tuple(inp[pos:pos + len(left)]) != tuple(left):
                    continue
                if mode == "tree":
                    expanded = []
                    valid = True
                    for t in right:
                        match = next(
                            (rr for ll, rr in rules if ll[0] == t), None
                        )
                        if match is None:
                            valid = False
                            break
                        expanded.extend(match)
                    if not valid:
                        continue
                    right = expanded
                result.extend(right)
                pos += len(left)
                found = True
                break

        if not found or len(result) > limit:
            return None
    return tuple(result), penalty


def _normal_goal(board, maps, failed):
    rules = [
        (tuple(map(_token, left)), tuple(map(_token, right)))
        for left, right in board["rules"]
    ]
    inp = tuple(map(_token, board["inp"]))
    current = tuple(_token(g[0]) for g in board["editable"])
    best = None
    best_score = float("inf")

    models = [("plain", ()), ("tree", ())]
    models += [("double", p) for p in _configurations(rules)]

    for mode, pairs in models:
        translated = _translate(inp, rules, mode, pairs, len(current))
        if translated is None:
            continue
        prefix, penalty = translated
        if any(a[0] != b[0] for a, b in zip(prefix, current)):
            continue
        target = prefix + current[len(prefix):]
        if target in failed:
            continue
        score = _target_cost(current, target, board["selected"], maps)
        score += 6 * (len(current) - len(prefix))
        score += 0.8 * len(pairs) + penalty
        if score < best_score:
            best_score, best = score, target
    return best


# Weighted union-find represented entirely by ordinary lists.
def _uf_new(n):
    return [list(range(n)), [0] * n]


def _uf_copy(u):
    return [u[0][:], u[1][:]]


def _uf_find(u, x):
    parent, weight = u
    offset = 0
    while parent[x] != x:
        offset += weight[x]
        x = parent[x]
    return x, offset % 7


def _uf_bind(u, a, b):
    # Terms contain (alphabet, current symbol number, edit variable).
    if a[0] != b[0]:
        return False
    ra, oa = _uf_find(u, a[2])
    rb, ob = _uf_find(u, b[2])
    difference = (ob + b[1] - oa - a[1]) % 7
    if ra == rb:
        return difference == 0
    if ra < rb:
        u[0][rb] = ra
        u[1][rb] = (-difference) % 7
    else:
        u[0][ra] = rb
        u[1][ra] = difference
    return True


def _uf_bind_sequences(u, a, b):
    return len(a) == len(b) and all(
        _uf_bind(u, x, y) for x, y in zip(a, b)
    )


def _uf_key(u):
    return tuple(_uf_find(u, i) for i in range(len(u[0])))


def _alter_goal(board, maps, failed, deadline):
    groups = board["editable"]
    nv = len(groups)
    constant = nv
    inverse = {c: {v: m for m, v in d.items()} for c, d in maps.items()}

    def term(cell, var):
        return (cell["c"], maps[cell["c"]][cell["m"]], var)

    rules = []
    for i, (left, right) in enumerate(board["rules"]):
        rules.append((
            tuple(term(c, 2 * i) for c in left),
            tuple(term(c, 2 * i + 1) for c in right),
        ))
    inp = tuple(term(c, constant) for c in board["inp"])
    out = tuple(term(c, constant) for c in board["out"])
    literal_in = tuple((t[0], t[1]) for t in inp)
    literal_out = tuple((t[0], t[1]) for t in out)
    current = tuple(_token(g[0]) for g in groups)
    first_terms = [term(g[0], i) for i, g in enumerate(groups)]

    best_target = None
    best_score = float("inf")

    def run_model(mode, pairs, seconds, node_limit, leaf_limit):
        nonlocal best_target, best_score
        end = min(deadline, time.monotonic() + seconds)
        effective = _effective(rules, pairs) if mode == "double" else None
        seen = set()
        nodes = 0
        leaves = 0

        def stopped():
            return (
                nodes >= node_limit
                or leaves >= leaf_limit
                or time.monotonic() >= end
            )

        def leaf(uf):
            nonlocal leaves, best_target, best_score
            leaves += 1
            locations = [_uf_find(uf, i) for i in range(nv)]
            cr, co = _uf_find(uf, constant)
            fixed = (-co) % 7
            members = {}
            for i, (root, offset) in enumerate(locations):
                members.setdefault(root, []).append((i, offset))

            free_roots = []
            choices = []
            for root, entries in members.items():
                if root == cr:
                    continue
                opts = []
                for value in range(7):
                    cost = sum(
                        min(
                            (value + offset) % 7,
                            (-(value + offset)) % 7,
                        )
                        for _, offset in entries
                    )
                    opts.append((cost, value))
                opts.sort()
                free_roots.append(root)
                choices.append(opts)

            start = (0,) * len(choices)
            heap = [(sum(c[0][0] for c in choices), start)]
            visited = {start}
            valid_at = None
            trials = 0

            while heap and trials < 80 and time.monotonic() < end:
                edit_lower, ranks = heapq.heappop(heap)
                trials += 1
                if valid_at is not None and edit_lower > valid_at + 2:
                    break

                root_values = {cr: fixed}
                for k, root in enumerate(free_roots):
                    root_values[root] = choices[k][ranks[k]][1]
                shifts = [
                    (root_values[root] + offset) % 7
                    for root, offset in locations
                ]

                literals = []
                for left, right in rules:
                    def substitute(seq):
                        return tuple(
                            (t[0], (t[1] + shifts[t[2]]) % 7)
                            for t in seq
                        )
                    literals.append((substitute(left), substitute(right)))

                prediction = _translate(
                    literal_in, literals, mode, pairs, len(literal_out)
                )
                if prediction is not None:
                    predicted, penalty = prediction
                    if predicted == literal_out[:len(predicted)]:
                        try:
                            target = tuple(
                                (
                                    t[0],
                                    inverse[t[0]][(t[1] + shifts[i]) % 7],
                                )
                                for i, t in enumerate(first_terms)
                            )
                        except KeyError:
                            target = None

                        if target is not None and target not in failed:
                            score = _target_cost(
                                current, target, board["selected"], maps
                            )
                            score += 6 * (len(out) - len(predicted))
                            score += 0.8 * len(pairs) + penalty
                            if score < best_score:
                                best_score, best_target = score, target
                            if valid_at is None:
                                valid_at = edit_lower

                for k in range(len(choices)):
                    if ranks[k] == 6:
                        continue
                    nr = list(ranks)
                    nr[k] += 1
                    nr = tuple(nr)
                    if nr in visited:
                        continue
                    visited.add(nr)
                    nc = (
                        edit_lower
                        - choices[k][ranks[k]][0]
                        + choices[k][nr[k]][0]
                    )
                    heapq.heappush(heap, (nc, nr))

        def dfs(pi, po, uf):
            nonlocal nodes
            nodes += 1
            if stopped():
                return
            if pi >= len(inp):
                leaf(uf)
                return
            key = (pi, po, _uf_key(uf))
            if key in seen:
                return
            seen.add(key)

            if mode == "double":
                for original_left, left, middle in effective:
                    if pi + len(original_left) > len(inp):
                        continue
                    u = _uf_copy(uf)
                    if not _uf_bind_sequences(
                        u, original_left, inp[pi:pi + len(original_left)]
                    ):
                        continue
                    for _, second_left, right in effective:
                        if po + len(right) > len(out):
                            continue
                        v = _uf_copy(u)
                        if not all(
                            _uf_bind(v, a, b)
                            for a, b in zip(middle, second_left)
                        ):
                            continue
                        if _uf_bind_sequences(
                            v, right, out[po:po + len(right)]
                        ):
                            dfs(pi + len(left), po + len(right), v)
                        if stopped():
                            return
                return

            for left, right in rules:
                if pi + len(left) > len(inp):
                    continue
                u = _uf_copy(uf)
                if not _uf_bind_sequences(
                    u, left, inp[pi:pi + len(left)]
                ):
                    continue

                if mode == "plain":
                    if po + len(right) > len(out):
                        continue
                    if _uf_bind_sequences(
                        u, right, out[po:po + len(right)]
                    ):
                        dfs(pi + len(left), po + len(right), u)
                else:
                    def expand(k, q, state):
                        if stopped():
                            return
                        if k == len(right):
                            dfs(pi + len(left), q, state)
                            return
                        for second_left, second_right in rules:
                            if q + len(second_right) > len(out):
                                continue
                            v = _uf_copy(state)
                            if not _uf_bind(v, right[k], second_left[0]):
                                continue
                            if _uf_bind_sequences(
                                v, second_right, out[q:q + len(second_right)]
                            ):
                                expand(k + 1, q + len(second_right), v)
                    expand(0, po, u)
                if stopped():
                    return

        dfs(0, 0, _uf_new(nv + 1))

    run_model("plain", (), 1.8, 18000, 220)
    if best_target is not None:
        return best_target

    run_model("tree", (), 1.25, 12000, 100)
    run_model("double", (), 1.25, 12000, 100)

    for pairs in _configurations(rules)[1:]:
        if time.monotonic() >= deadline:
            break
        run_model("double", pairs, 0.065, 1800, 24)

    return best_target


def act(obs, available, memory):
    started = time.monotonic()
    available = set(available)
    if "maps" not in memory:
        memory["maps"] = _initial_maps()
    maps = memory["maps"]

    def emit(action, board=None, direction=None):
        if action not in available:
            action = next(
                (
                    a for a in ("ACTION2", "ACTION1", "ACTION4", "ACTION3")
                    if a in available
                ),
                next(iter(available), "RESET"),
            )
        if board is not None and direction is not None:
            index = board["selected"]
            c = board["editable"][index][0]
            memory["last_edit"] = (
                c["c"], c["m"], direction, index, board["alter"]
            )
        return action, None, None

    board = _parse(obs, memory.get("mode", False))
    if board is None:
        return emit("ACTION2")

    if memory.get("board_key") != board["key"]:
        for key in ("active", "failed", "learning", "last_edit"):
            memory.pop(key, None)
        memory["board_key"] = board["key"]

    memory["mode"] = board["alter"]
    memory.setdefault("failed", [])
    failed = set(memory["failed"])
    groups = board["editable"]
    selected = board["selected"]

    last = memory.pop("last_edit", None)
    if last is not None:
        color, old_mask, direction, index, old_mode = last
        if old_mode == board["alter"] and index < len(groups):
            c = groups[index][0]
            if c["c"] == color and c["m"] != old_mask:
                lookup = maps.setdefault(color, {})
                new_mask = c["m"]
                if old_mask in lookup:
                    value = (lookup[old_mask] + direction) % 7
                    if new_mask not in lookup:
                        lookup[new_mask] = value
                elif new_mask in lookup:
                    lookup[old_mask] = (lookup[new_mask] - direction) % 7
                elif not lookup:
                    lookup[old_mask] = 0
                    lookup[new_mask] = direction % 7

    if board["alter"] and "active" not in memory:
        all_cells = (
            board["inp"]
            + board["out"]
            + [c for pair in board["rules"] for g in pair for c in g]
        )
        missing = {
            c["c"] for c in all_cells
            if c["m"] not in maps.get(c["c"], {})
        }
        if "learning" not in memory and missing:
            index = next(
                (
                    i for i, g in enumerate(groups)
                    if any(c["c"] in missing for c in g)
                ),
                None,
            )
            if index is not None:
                families = sorted(
                    {c["c"] for c in groups[index]} & missing
                )
                memory["learning"] = {
                    "index": index,
                    "families": families,
                    "count": 0,
                    "initialized": False,
                }

        learning = memory.get("learning")
        if learning is not None:
            index = learning["index"]
            if selected != index:
                direction = _route(len(groups), selected, [index])[1]
                return emit("ACTION4" if direction > 0 else "ACTION3")

            if not learning["initialized"]:
                for color in learning["families"]:
                    maps[color] = {}
                learning["initialized"] = True

            for color in learning["families"]:
                c = next(c for c in groups[index] if c["c"] == color)
                maps[color][c["m"]] = learning["count"] % 7

            if learning["count"] < 7:
                learning["count"] += 1
                return emit("ACTION2", board, 1)
            memory.pop("learning", None)

    current = tuple(_token(g[0]) for g in groups)
    active = memory.get("active")

    if active is not None:
        target = active["target"]
        if len(target) != len(current):
            memory.pop("active", None)
            active = None
        elif active.get("undo"):
            active["undo"] = False
            return emit("ACTION2", board, 1)
        elif current == target and active.get("touched"):
            memory["failed"].append(target)
            memory["failed"] = memory["failed"][-150:]
            failed.add(target)
            memory.pop("active", None)
            active = None

    if active is None:
        if board["alter"]:
            known = all(
                c["m"] in maps.get(c["c"], {})
                for c in (
                    board["inp"]
                    + board["out"]
                    + [c for p in board["rules"] for g in p for c in g]
                )
            )
            target = (
                _alter_goal(board, maps, failed, started + 7.0)
                if known else None
            )
        else:
            target = _normal_goal(board, maps, failed)

        if target is None:
            count = memory.get("fallback_count", 0)
            memory["fallback_count"] = count + 1
            if count % 8 == 7:
                return emit("ACTION4")
            return emit("ACTION2", board, 1)

        active = {"target": target, "touched": False, "undo": False}
        memory["active"] = active

    target = active["target"]
    dirty = [
        i for i, (a, b) in enumerate(zip(current, target))
        if a != b
    ]

    if not dirty:
        active["touched"] = True
        active["undo"] = True
        return emit("ACTION1", board, -1)

    if selected not in dirty:
        direction = _route(len(groups), selected, dirty)[1]
        return emit("ACTION4" if direction > 0 else "ACTION3")

    a, b = current[selected], target[selected]
    lookup = maps.get(a[0], {})
    direction = 1
    if a[1] in lookup and b[1] in lookup:
        distance = (lookup[b[1]] - lookup[a[1]]) % 7
        direction = 1 if distance <= 3 else -1

    active["touched"] = True
    return emit("ACTION2" if direction > 0 else "ACTION1", board, direction)

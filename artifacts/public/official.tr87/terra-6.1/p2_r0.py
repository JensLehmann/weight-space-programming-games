from functools import lru_cache


def _canonical(rows):
    """Rotation-invariant representation of a 5 by 5 binary symbol."""
    a = tuple(tuple(int(v) for v in row) for row in rows)
    variants = []
    for _ in range(4):
        variants.append("".join(str(v) for row in a for v in row))
        a = tuple(tuple(a[4 - x][y] for x in range(5)) for y in range(5))
    return min(variants)


# The known families. The other families, if present, are learned in play.
_FAMILIES = {
    10: [
        ["00100", "00100", "01110", "00100", "11111"],
        ["00001", "00101", "11111", "10100", "10000"],
        ["11111", "10100", "10000", "10100", "11111"],
        ["00100", "11111", "10101", "10101", "00100"],
        ["10001", "11111", "00100", "11111", "10001"],
        ["11111", "00100", "11100", "10000", "11111"],
        ["10001", "11111", "10001", "10001", "11011"],
    ],
    7: [
        ["10000", "11110", "10010", "10010", "11111"],
        ["11111", "10001", "11101", "10101", "11111"],
        ["00111", "00101", "11111", "10100", "11100"],
        ["01110", "01010", "11111", "10001", "11111"],
        ["11111", "10001", "10001", "11111", "00100"],
        ["11110", "10011", "10001", "11001", "01111"],
        ["00100", "11111", "10101", "11111", "00100"],
    ],
    11: [
        ["10101", "10000", "10101", "10000", "11111"],
        ["11101", "10000", "00000", "00001", "10111"],
        ["10111", "00100", "00100", "00100", "11101"],
        ["00100", "10101", "00100", "10101", "00100"],
        ["11011", "10001", "10101", "10001", "11011"],
        ["10111", "10101", "11101", "00000", "10101"],
        ["10001", "00000", "10001", "11011", "01010"],
    ],
}


def _initial_maps():
    result = {}
    for color, family in _FAMILIES.items():
        result[color] = {
            _canonical([[int(ch) for ch in row] for row in symbol]): i
            for i, symbol in enumerate(family)
        }
    return result


def _read_board(obs):
    """Return tiles, partitioned into the rule area and the two message rows."""
    split = None
    for y in range(2, 60):
        if sum(int(obs[y, x] == 3) for x in range(64)) >= 60:
            split = y
            break
    if split is None:
        split = 34

    candidates = {}
    for y in range(57):
        row_tiles = []
        x = 0
        while x < 64:
            c = int(obs[y, x])
            if c in (2, 3, 5) or int(obs[y + 6, x]) != c:
                x += 1
                continue
            end = x + 1
            while (end < 64 and int(obs[y, end]) == c
                   and int(obs[y + 6, end]) == c):
                end += 1
            length = end - x
            if length >= 7 and length % 7 == 0:
                for left in range(x, end, 7):
                    if not all(
                        int(obs[yy, left]) == c
                        and int(obs[yy, left + 6]) == c
                        for yy in range(y, y + 7)
                    ):
                        continue
                    interior = [
                        [int(obs[y + 1 + dy, left + 1 + dx]) for dx in range(5)]
                        for dy in range(5)
                    ]
                    if not all(v in (c, 5) for row in interior for v in row):
                        continue
                    bits = [[int(v == 5) for v in row] for row in interior]
                    if not any(any(row) for row in bits):
                        continue
                    row_tiles.append({
                        "x": left,
                        "y": y,
                        "color": c,
                        "key": (c, _canonical(bits)),
                    })
            x = end
        if row_tiles:
            candidates[y] = row_tiles

    # Reject spurious overlapping 7-pixel rows.
    chosen = []
    for y in sorted(candidates, key=lambda yy: (-len(candidates[yy]), yy)):
        if all(abs(y - other) >= 7 for other in chosen):
            chosen.append(y)
    tiles = []
    for y in sorted(chosen):
        tiles.extend(sorted(candidates[y], key=lambda t: t["x"]))

    top = [t for t in tiles if t["y"] < split]
    bottom = [t for t in tiles if t["y"] >= split]
    bottom_ys = sorted(set(t["y"] for t in bottom))
    if len(bottom_ys) < 2:
        return None

    message_left = sorted(
        [t for t in bottom if t["y"] == bottom_ys[0]], key=lambda t: t["x"]
    )
    message_right = sorted(
        [t for t in bottom if t["y"] == bottom_ys[1]], key=lambda t: t["x"]
    )

    groups = []
    for y in sorted(set(t["y"] for t in top)):
        row = sorted([t for t in top if t["y"] == y], key=lambda t: t["x"])
        row_groups = []
        for tile in row:
            if row_groups and tile["x"] == row_groups[-1][-1]["x"] + 7:
                row_groups[-1].append(tile)
            else:
                row_groups.append([tile])
        if len(row_groups) % 2:
            return None
        groups.extend(row_groups)

    if not groups or len(groups) % 2:
        return None
    rules = [(groups[i], groups[i + 1]) for i in range(0, len(groups), 2)]

    def marked(tile):
        y, x = tile["y"] - 3, tile["x"] + 1
        return (
            0 <= y < 64 and 0 <= x and x + 5 <= 64
            and all(int(obs[y, xx]) == 0 for xx in range(x, x + 5))
        )

    selected_top = next((i for i, group in enumerate(groups)
                         if marked(group[0])), None)
    selected_bottom = next((i for i, tile in enumerate(message_right)
                            if marked(tile)), None)
    if selected_top is not None:
        altered = True
        selected = selected_top
        controls = groups
    elif selected_bottom is not None:
        altered = False
        selected = selected_bottom
        controls = [[t] for t in message_right]
    else:
        return None

    return {
        "rules": rules,
        "left": message_left,
        "right": message_right,
        "altered": altered,
        "selected": selected,
        "controls": controls,
    }


def _keys(tiles):
    return tuple(t["key"] for t in tiles)


def _matches(sequence, pos, pattern):
    return sequence[pos:pos + len(pattern)] == pattern


def _navigation(n, start, mask):
    """Shortest walk on a circular selector visiting all requested indices."""
    if not mask:
        return []

    @lru_cache(None)
    def search():
        best = None
        for direction in (-1, 1):
            for turn_after in range(n):
                pos = start
                seen = (1 << pos) & mask
                route = []
                for _ in range(turn_after):
                    pos = (pos + direction) % n
                    route.append(direction)
                    seen |= (1 << pos) & mask
                if seen != mask:
                    for _ in range(n):
                        pos = (pos - direction) % n
                        route.append(-direction)
                        seen |= (1 << pos) & mask
                        if seen == mask:
                            break
                if seen == mask and (best is None or len(route) < len(best)):
                    best = route
        return best or []
    return search()


def _make_plan(shifts, start):
    n = len(shifts)
    mask = sum(1 << i for i, d in enumerate(shifts) if d % 7)
    route = _navigation(n, start, mask)
    plan = []
    done = set()
    pos = start

    def rotate_here():
        if pos in done:
            return
        done.add(pos)
        d = shifts[pos] % 7
        if d <= 3:
            plan.extend(["ACTION2"] * d)
        else:
            plan.extend(["ACTION1"] * (7 - d))

    rotate_here()
    for step in route:
        plan.append("ACTION4" if step == 1 else "ACTION3")
        pos = (pos + step) % n
        rotate_here()
    return plan


def _move_to(current, target, n):
    right = (target - current) % n
    left = (current - target) % n
    return "ACTION4" if right <= left else "ACTION3"


def _advance_key(key, delta, maps, inverse):
    color, signature = key
    mapping = maps.get(color, {})
    if signature not in mapping or color not in inverse:
        return None
    new_signature = inverse[color].get((mapping[signature] + delta) % 7)
    return None if new_signature is None else (color, new_signature)


def _standard_candidates(board):
    """Generate direct, tree, and possible two-stage translations."""
    rules = [(_keys(a), _keys(b)) for a, b in board["rules"]]
    source = _keys(board["left"])
    destination = _keys(board["right"])
    nout = len(destination)
    results = []
    seen = set()

    def add(output):
        if len(output) > nout:
            return
        if any(output[i][0] != destination[i][0] for i in range(len(output))):
            return
        target = tuple(output) + destination[len(output):]
        if target not in seen:
            seen.add(target)
            results.append(target)

    def straightforward(tree):
        pos = 0
        output = []
        while pos < len(source):
            found = False
            for lhs, rhs in rules:
                if not _matches(source, pos, lhs):
                    continue
                found = True
                if tree:
                    translated = []
                    for symbol in rhs:
                        for lhs2, rhs2 in rules:
                            if lhs2 and lhs2[0] == symbol:
                                translated.extend(rhs2)
                                break
                        else:
                            return
                    output.extend(translated)
                else:
                    output.extend(rhs)
                pos += len(lhs)
                break
            if not found or len(output) > nout:
                return
        add(output)

    straightforward(False)
    straightforward(True)

    # In double-translation levels, hidden markers can join pairs of rules.
    # Enumerate plausible joined rules, then compose them.
    variants = []
    for i, (lhs, rhs) in enumerate(rules):
        variants.append((lhs, rhs))
        for j, (lhs2, rhs2) in enumerate(rules):
            if i != j:
                variants.append((lhs + lhs2, rhs + rhs2))

    options = {}
    for lhs, mid in variants:
        for lhs2, rhs2 in variants:
            if all(a == b for a, b in zip(mid, lhs2)):
                options.setdefault(lhs, set()).add(rhs2)

    ordered = list(options.items())
    ordered.sort(key=lambda item: -len(item[0]))
    limit = 800

    def compose(pos, output):
        if len(results) >= limit or len(output) > nout:
            return
        if any(output[i][0] != destination[i][0] for i in range(len(output))):
            return
        if pos == len(source):
            add(output)
            return
        for lhs, outputs in ordered:
            if _matches(source, pos, lhs):
                for rhs in outputs:
                    compose(pos + len(lhs), output + rhs)
                    if len(results) >= limit:
                        return

    compose(0, ())
    return results


def _standard_plan(board, maps, inverse, failed):
    current = _keys(board["right"])
    best = None
    for target in _standard_candidates(board):
        attempt = ("standard", target)
        if attempt in failed:
            continue
        shifts = []
        for old, new in zip(current, target):
            if old[0] != new[0]:
                break
            mapping = maps.get(old[0], {})
            if old[1] not in mapping or new[1] not in mapping:
                break
            shifts.append((mapping[new[1]] - mapping[old[1]]) % 7)
        else:
            plan = _make_plan(shifts, board["selected"])
            if not plan:
                plan = ["ACTION1", "ACTION2"]
            if best is None or len(plan) < len(best[0]):
                best = (plan, attempt)
    return best


def _altered_plan(board, maps, inverse, failed):
    groups = board["controls"]
    source = _keys(board["left"])
    destination = _keys(board["right"])
    raw_rules = [(_keys(a), _keys(b)) for a, b in board["rules"]]
    n = len(groups)

    shifted = []
    for group in groups:
        base = _keys(group)
        choices = []
        for delta in range(7):
            value = tuple(_advance_key(k, delta, maps, inverse) for k in base)
            choices.append(value)
        shifted.append(choices)

    best = None
    completed = 0
    max_completed = 2500

    def verify(shifts):
        pos_in = pos_out = 0
        while pos_in < len(source):
            for r in range(len(raw_rules)):
                lhs = shifted[2 * r][shifts[2 * r]]
                rhs = shifted[2 * r + 1][shifts[2 * r + 1]]
                if _matches(source, pos_in, lhs):
                    if not _matches(destination, pos_out, rhs):
                        return False
                    pos_in += len(lhs)
                    pos_out += len(rhs)
                    break
            else:
                return False
        return True

    def visit(pos_in, pos_out, assignment):
        nonlocal best, completed
        if completed >= max_completed:
            return
        if pos_in == len(source):
            completed += 1
            shifts = [assignment.get(i, 0) for i in range(n)]
            if not verify(shifts):
                return
            target = tuple(shifted[i][shifts[i]] for i in range(n))
            attempt = ("altered", target)
            if attempt in failed:
                return
            plan = _make_plan(shifts, board["selected"])
            if not plan:
                plan = ["ACTION1", "ACTION2"]
            if best is None or len(plan) < len(best[0]):
                best = (plan, attempt)
            return

        for r in range(len(raw_rules)):
            li, ri = 2 * r, 2 * r + 1
            left_choices = [assignment[li]] if li in assignment else range(7)
            right_choices = [assignment[ri]] if ri in assignment else range(7)
            for dl in left_choices:
                lhs = shifted[li][dl]
                if None in lhs or not _matches(source, pos_in, lhs):
                    continue
                for dr in right_choices:
                    rhs = shifted[ri][dr]
                    if None in rhs or not _matches(destination, pos_out, rhs):
                        continue
                    new_assignment = dict(assignment)
                    new_assignment[li] = dl
                    new_assignment[ri] = dr
                    visit(pos_in + len(lhs), pos_out + len(rhs), new_assignment)
                    if completed >= max_completed:
                        return

    visit(0, 0, {})
    return best


def _emit_queued(memory):
    action = memory["queue"].pop(0)
    if not memory["queue"]:
        memory["check"] = memory.pop("attempt")
    return action, None, None


def act(obs, available, memory):
    if "maps" not in memory:
        memory["maps"] = _initial_maps()
        memory["failed"] = set()

    if memory.get("queue"):
        return _emit_queued(memory)

    if "check" in memory:
        # If this call occurred, the preceding complete plan did not win.
        memory["failed"].add(memory.pop("check"))

    board = _read_board(obs)
    if board is None:
        action = "ACTION4" if "ACTION4" in available else available[0]
        return action, None, None

    maps = memory["maps"]
    controls = board["controls"]
    selected = board["selected"]

    # Continue an experiment to learn an unfamiliar seven-symbol cycle.
    learner = memory.get("learner")
    if learner is not None:
        target = learner["control"]
        if target >= len(controls):
            memory.pop("learner", None)
        elif selected != target:
            return _move_to(selected, target, len(controls)), None, None
        else:
            tile = controls[target][learner["offset"]]
            signature = tile["key"][1]
            if learner["waiting"]:
                learner["seen"].append(signature)
                learner["waiting"] = False
            if len(learner["seen"]) >= 7:
                maps[learner["color"]] = {
                    sig: i for i, sig in enumerate(learner["seen"])
                }
                memory.pop("learner", None)
            else:
                learner["waiting"] = True
                return "ACTION2", None, None

    # Only mutable families need a cycle map.
    for ci, group in enumerate(controls):
        for offset, tile in enumerate(group):
            color, signature = tile["key"]
            if signature not in maps.get(color, {}) or len(maps.get(color, {})) < 7:
                memory["learner"] = {
                    "control": ci,
                    "offset": offset,
                    "color": color,
                    "seen": [signature],
                    "waiting": False,
                }
                if selected != ci:
                    return _move_to(selected, ci, len(controls)), None, None
                memory["learner"]["waiting"] = True
                return "ACTION2", None, None

    inverse = {
        color: {index: signature for signature, index in mapping.items()}
        for color, mapping in maps.items()
    }

    if board["altered"]:
        answer = _altered_plan(board, maps, inverse, memory["failed"])
    else:
        answer = _standard_plan(board, maps, inverse, memory["failed"])

    if answer is None:
        # Avoid an infinite loop if a speculative model was exhausted.
        if memory["failed"]:
            memory["failed"].clear()
            if board["altered"]:
                answer = _altered_plan(board, maps, inverse, memory["failed"])
            else:
                answer = _standard_plan(board, maps, inverse, memory["failed"])
        if answer is None:
            return "ACTION2", None, None

    memory["queue"], memory["attempt"] = answer
    return _emit_queued(memory)

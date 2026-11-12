from collections import deque


_TEMPLATES = {
    10: [
        "0010000100011100010011111",
        "0000100101111111010010000",
        "1111110100100001010011111",
        "0010011111101011010100100",
        "1000111111001001111110001",
        "1111100100111001000011111",
        "1000111111100011000111011",
    ],
    7: [
        "1000011110100101001011111",
        "1111110001111011010111111",
        "0011100101111111010011100",
        "0111001010111111000111111",
        "1111110001100011111100100",
        "1111010011100011100101111",
        "0010011111101011111100100",
    ],
    11: [
        "1010110000101011000011111",
        "1110110000000000000110111",
        "1011100100001000010011101",
        "0010010101001001010100100",
        "1101110001101011000111011",
        "1011110101111010000010101",
        "1000100000100011101101010",
    ],
}


def _freeze(value):
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(x) for x in value)
    return value


def _canonical(bits):
    values = []
    current = bits
    for _ in range(4):
        values.append(current)
        rotated = 0
        for y in range(5):
            for x in range(5):
                if current & (1 << (5 * y + x)):
                    rotated |= 1 << (5 * x + 4 - y)
        current = rotated
    return min(values)


def _template_graphs():
    graphs = {}
    for color, patterns in _TEMPLATES.items():
        keys = []
        for pattern in patterns:
            bits = 0
            for i, ch in enumerate(pattern):
                if ch == "1":
                    bits |= 1 << i
            keys.append(_canonical(bits))
        graphs[color] = {
            keys[i]: keys[(i + 1) % len(keys)] for i in range(len(keys))
        }
    return graphs


_BASE_GRAPHS = _template_graphs()


def _read_tiles(obs):
    candidates = []
    excluded = {0, 2, 3, 5}

    for y in range(57):
        for x in range(57):
            color = int(obs[y, x])
            if color in excluded:
                continue

            good = True
            for k in range(7):
                if int(obs[y, x + k]) not in (color, 0):
                    good = False
                    break
                if int(obs[y + 6, x + k]) not in (color, 0):
                    good = False
                    break
                if int(obs[y + k, x]) not in (color, 0):
                    good = False
                    break
                if int(obs[y + k, x + 6]) not in (color, 0):
                    good = False
                    break
            if not good:
                continue

            bits = 0
            count = 0
            for dy in range(5):
                for dx in range(5):
                    v = int(obs[y + 1 + dy, x + 1 + dx])
                    if v == 5:
                        bits |= 1 << (5 * dy + dx)
                        count += 1
                    elif v not in (color, 0):
                        good = False
                        break
                if not good:
                    break

            if good and count >= 2:
                candidates.append((y, x, color, _canonical(bits)))

    tiles = []
    for tile in sorted(candidates):
        y, x = tile[:2]
        if any(abs(y - t[0]) < 7 and abs(x - t[1]) < 7 for t in tiles):
            continue
        tiles.append(tile)
    return tiles


def _parse(obs):
    tiles = _read_tiles(obs)
    boundary = None
    for y in range(63):
        if all(int(obs[y, x]) == 3 for x in range(64)):
            boundary = y
            break
    if boundary is None:
        return None

    upper = [t for t in tiles if t[0] < boundary]
    lower = [t for t in tiles if t[0] > boundary]
    lower_rows = sorted(set(t[0] for t in lower))
    if len(lower_rows) < 2:
        return None

    source = sorted(
        [t for t in lower if t[0] == lower_rows[0]], key=lambda t: t[1]
    )
    target = sorted(
        [t for t in lower if t[0] == lower_rows[1]], key=lambda t: t[1]
    )
    if not source or not target:
        return None

    rules = []
    for y in sorted(set(t[0] for t in upper)):
        row = sorted([t for t in upper if t[0] == y], key=lambda t: t[1])
        groups = []
        for tile in row:
            if groups and tile[1] == groups[-1][-1][1] + 7:
                groups[-1].append(tile)
            else:
                groups.append([tile])

        i = 0
        while i + 1 < len(groups):
            left, right = groups[i], groups[i + 1]
            end = left[-1][1] + 7
            start = right[0][1]
            connected = (
                start > end
                and start - end <= 5
                and all(int(obs[y + 3, x]) == 3 for x in range(end, start))
            )
            if connected:
                rules.append((left, right))
                i += 2
            else:
                i += 1

    if not rules:
        return None

    sets = [group for rule in rules for group in rule]

    def has_cursor(tile):
        y, x = tile[:2]
        cy = y - 3
        if cy < 0 or x + 6 >= 64:
            return False
        return all(int(obs[cy, x + 1 + k]) == 0 for k in range(5))

    selected = None
    mode = None
    for i, group in enumerate(sets):
        if has_cursor(group[0]):
            selected = i
            mode = "rules"
            break

    if mode is None:
        for i, tile in enumerate(target):
            if has_cursor(tile):
                selected = i
                mode = "target"
                break

    if mode is None:
        return None

    return {
        "tiles": tiles,
        "source": source,
        "target": target,
        "rules": rules,
        "sets": sets,
        "mode": mode,
        "selected": selected,
    }


def _symbols(tiles):
    return tuple((t[2], t[3]) for t in tiles)


def _graphs(memory):
    graphs = {color: dict(edges) for color, edges in _BASE_GRAPHS.items()}
    for color, edges in memory.get("edges", {}).items():
        graph = graphs.setdefault(int(color), {})
        for a, b in edges.items():
            graph[int(a)] = int(b)
    return graphs


def _learn(memory, parsed):
    previous = memory.pop("previous_rotation", None)
    if not previous:
        return

    now = {(t[0], t[1], t[2]): t[3] for t in parsed["tiles"]}
    edges = memory.setdefault("edges", {})
    direction = previous["direction"]

    for y, x, color, old in previous["tiles"]:
        new = now.get((y, x, color))
        if new is None or old == new:
            continue
        a, b = (old, new) if direction == 1 else (new, old)
        edges.setdefault(str(color), {})[str(a)] = b


def _step_symbol(symbol, direction, graphs):
    color, key = symbol
    graph = graphs.get(color, {})
    if direction == 1:
        nxt = graph.get(key)
    else:
        nxt = next((a for a, b in graph.items() if b == key), None)
    return None if nxt is None else (color, nxt)


def _distance(a, b, graphs):
    if a == b:
        return (0, 0)
    if a[0] != b[0]:
        return None

    queue = deque([(a, 0, 0)])
    seen = {a}
    while queue:
        symbol, dist, first = queue.popleft()
        for direction in (-1, 1):
            nxt = _step_symbol(symbol, direction, graphs)
            if nxt is None or nxt in seen:
                continue
            first_direction = direction if dist == 0 else first
            if nxt == b:
                return (dist + 1, first_direction)
            seen.add(nxt)
            queue.append((nxt, dist + 1, first_direction))
    return None


def _route(n, selected, needed):
    needed = sorted(set(needed))
    if not needed or selected in needed:
        return (0, 0)

    best = None
    for cut in range(n):
        s = (selected - cut) % n
        positions = [(i - cut) % n for i in needed]
        lo, hi = min(positions), max(positions)
        span = hi - lo

        for distance, first_target in (
            (abs(s - lo) + span, lo),
            (abs(s - hi) + span, hi),
        ):
            if first_target == s:
                first_target = hi if first_target == lo else lo
            direction = 1 if first_target > s else -1
            candidate = (distance, direction)
            if best is None or candidate < best:
                best = candidate
    return best


def _direct_translation(source, rules, tree=False):
    result = []
    pos = 0

    while pos < len(source):
        for left, right in rules:
            if source[pos:pos + len(left)] != left:
                continue

            if tree:
                translated = []
                for symbol in right:
                    for next_left, next_right in rules:
                        if next_left and next_left[0] == symbol:
                            translated.extend(next_right)
                            break
                    else:
                        return None
                result.extend(translated)
            else:
                result.extend(right)

            pos += len(left)
            break
        else:
            return None

    return tuple(result)


def _double_candidates(source, target, rules, limit=2500):
    choices = []
    for i, (left, right) in enumerate(rules):
        choices.append((i, left, right))
        for j, (left2, right2) in enumerate(rules):
            if i != j:
                choices.append((i, left + left2, right + right2))

    results = set()
    calls = [0]

    def search(pos, out):
        calls[0] += 1
        if calls[0] > limit * 20 or len(results) >= limit:
            return

        if pos >= len(source):
            if len(out) <= len(target):
                results.add(tuple(out) + target[len(out):])
            return

        local = set()
        for i, combined_left, intermediate in choices:
            original_left = rules[i][0]
            if source[pos:pos + len(original_left)] != original_left:
                continue
            if pos + len(combined_left) > len(source):
                continue

            for _, next_left, next_right in choices:
                if not all(a == b for a, b in zip(intermediate, next_left)):
                    continue
                if len(out) + len(next_right) > len(target):
                    continue

                segment = target[len(out):len(out) + len(next_right)]
                if any(a[0] != b[0] for a, b in zip(next_right, segment)):
                    continue

                option = (len(combined_left), next_right)
                if option in local:
                    continue
                local.add(option)
                search(pos + len(combined_left), out + next_right)

    search(0, ())
    return results


def _target_plan(parsed, graphs, failed):
    source = _symbols(parsed["source"])
    target = _symbols(parsed["target"])
    rules = [(_symbols(a), _symbols(b)) for a, b in parsed["rules"]]
    candidates = []

    for tree in (False, True):
        result = _direct_translation(source, rules, tree)
        if result is not None and len(result) <= len(target):
            candidates.append(result + target[len(result):])

    def evaluate(candidate):
        if str(candidate) in failed or len(candidate) != len(target):
            return None

        moves = []
        for current, wanted in zip(target, candidate):
            distance = _distance(current, wanted, graphs)
            if distance is None:
                return None
            moves.append(distance)

        needed = [i for i, (d, _) in enumerate(moves) if d]
        travel, _ = _route(len(target), parsed["selected"], needed)
        cost = sum(d for d, _ in moves) + travel
        return (cost, candidate)

    plans = [p for p in (evaluate(c) for c in candidates) if p is not None]
    if not plans:
        for candidate in _double_candidates(source, target, rules):
            plan = evaluate(candidate)
            if plan is not None:
                plans.append(plan)

    if not plans:
        return None
    return min(plans, key=lambda p: (p[0], str(p[1])))[1]


def _group_variants(group, graphs):
    original = _symbols(group)
    variants = [(0, original)]

    for direction in (-1, 1):
        current = original
        for count in range(1, 4):
            next_symbols = []
            for symbol in current:
                nxt = _step_symbol(symbol, direction, graphs)
                if nxt is None:
                    break
                next_symbols.append(nxt)
            else:
                current = tuple(next_symbols)
                variants.append((direction * count, current))
                continue
            break

    unique = {}
    for shift, value in variants:
        if value not in unique or abs(shift) < abs(unique[value]):
            unique[value] = shift
    return [(shift, value) for value, shift in unique.items()]


def _rule_plan(parsed, graphs, failed):
    source = _symbols(parsed["source"])
    target = _symbols(parsed["target"])
    groups = parsed["sets"]
    variants = [_group_variants(group, graphs) for group in groups]
    count = len(groups)
    assignments = [None] * count
    best = [None]
    visits = [0]

    def validate(values):
        p = q = 0
        while p < len(source):
            for i in range(0, count, 2):
                left, right = values[i], values[i + 1]
                if source[p:p + len(left)] != left:
                    continue
                if target[q:q + len(right)] != right:
                    return False
                p += len(left)
                q += len(right)
                break
            else:
                return False
        return True

    def consider():
        shifts = []
        values = []
        for i in range(count):
            choice = assignments[i]
            if choice is None:
                choice = (0, _symbols(groups[i]))
            shifts.append(choice[0])
            values.append(choice[1])

        signature = tuple(values)
        if str(signature) in failed or not validate(values):
            return

        needed = [i for i, shift in enumerate(shifts) if shift]
        travel, _ = _route(count, parsed["selected"], needed)
        cost = sum(abs(s) for s in shifts) + travel
        plan = (cost, signature)
        if best[0] is None or (cost, str(signature)) < (
            best[0][0], str(best[0][1])
        ):
            best[0] = plan

    def search(p, q):
        visits[0] += 1
        if visits[0] > 50000:
            return
        if p >= len(source):
            consider()
            return

        for i in range(0, count, 2):
            left_options = (
                [assignments[i]] if assignments[i] is not None else variants[i]
            )
            right_options = (
                [assignments[i + 1]]
                if assignments[i + 1] is not None
                else variants[i + 1]
            )

            for left_choice in left_options:
                left = left_choice[1]
                if source[p:p + len(left)] != left:
                    continue

                for right_choice in right_options:
                    right = right_choice[1]
                    if target[q:q + len(right)] != right:
                        continue

                    old_left = assignments[i]
                    old_right = assignments[i + 1]
                    assignments[i] = left_choice
                    assignments[i + 1] = right_choice
                    search(p + len(left), q + len(right))
                    assignments[i] = old_left
                    assignments[i + 1] = old_right

    search(0, 0)
    return None if best[0] is None else best[0][1]


def _moves_to_plan(parsed, graphs, signature):
    if parsed["mode"] == "target":
        current = _symbols(parsed["target"])
        if len(current) != len(signature):
            return None

        moves = []
        for a, b in zip(current, signature):
            distance = _distance(a, b, graphs)
            if distance is None:
                return None
            moves.append(distance)
        return moves

    groups = parsed["sets"]
    if len(groups) != len(signature):
        return None

    moves = []
    for group, wanted in zip(groups, signature):
        options = [
            shift for shift, value in _group_variants(group, graphs)
            if value == wanted
        ]
        if not options:
            return None
        shift = min(options, key=lambda s: (abs(s), s))
        moves.append((abs(shift), 0 if shift == 0 else (1 if shift > 0 else -1)))
    return moves


def act(obs, available, memory):
    parsed = _parse(obs)
    if parsed is None:
        action = "ACTION4" if "ACTION4" in available else available[0]
        return (action, None, None)

    _learn(memory, parsed)
    graphs = _graphs(memory)
    mode = parsed["mode"]

    current_signature = (
        _symbols(parsed["target"])
        if mode == "target"
        else tuple(_symbols(group) for group in parsed["sets"])
    )
    failed = memory.setdefault("failed", [])
    attempted = memory.pop("attempted", None)
    if attempted and attempted[0] == mode and attempted[1] == str(current_signature):
        if attempted[1] not in failed:
            failed.append(attempted[1])
        memory.pop("plan", None)

    def emit(action, attempt=None):
        if action in ("ACTION1", "ACTION2"):
            memory["previous_rotation"] = {
                "direction": -1 if action == "ACTION1" else 1,
                "tiles": [list(t) for t in parsed["tiles"]],
            }
        if attempt is not None:
            memory["attempted"] = [mode, str(attempt)]
        return (action, None, None)

    saved = memory.get("plan")
    signature = None
    moves = None

    if saved and saved.get("mode") == mode:
        signature = _freeze(saved.get("signature"))
        if str(signature) not in failed:
            moves = _moves_to_plan(parsed, graphs, signature)

    if moves is None:
        memory.pop("plan", None)
        if mode == "target":
            signature = _target_plan(parsed, graphs, failed)
        else:
            signature = _rule_plan(parsed, graphs, failed)

        if signature is not None:
            moves = _moves_to_plan(parsed, graphs, signature)
            if moves is not None:
                memory["plan"] = {"mode": mode, "signature": signature}

    if moves is not None:
        selected = parsed["selected"]
        remaining = sum(distance for distance, _ in moves)

        if remaining == 0:
            # The simulator checks completion after a symbol change. Move away
            # once, then the committed plan will bring the symbol back.
            return emit("ACTION2")

        distance, direction = moves[selected]
        if distance:
            action = "ACTION1" if direction < 0 else "ACTION2"
            attempt = signature if remaining == 1 else None
            return emit(action, attempt)

        needed = [i for i, (distance, _) in enumerate(moves) if distance]
        _, direction = _route(len(moves), selected, needed)
        return emit("ACTION3" if direction < 0 else "ACTION4")

    # Explore an unfamiliar symbol cycle when no complete plan is available.
    editable = (
        [[tile] for tile in parsed["target"]]
        if mode == "target"
        else parsed["sets"]
    )
    selected = parsed["selected"]
    unknown = []
    for i, group in enumerate(editable):
        colors = {tile[2] for tile in group}
        if any(len(graphs.get(color, {})) < 7 for color in colors):
            unknown.append(i)

    if unknown:
        if selected in unknown:
            return emit("ACTION2")
        _, direction = _route(len(editable), selected, unknown)
        return emit("ACTION3" if direction < 0 else "ACTION4")

    return emit("ACTION2")

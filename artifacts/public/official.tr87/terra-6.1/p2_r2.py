from collections import deque


_PATTERNS = {
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


def _canonical(rows):
    a = [list(r) for r in rows]
    forms = []
    for _ in range(4):
        n = 0
        for row in a:
            for v in row:
                n = (n << 1) | int(v)
        forms.append(n)
        a = [[a[4 - x][y] for x in range(5)] for y in range(5)]
    return min(forms)


def _key(color, rows):
    return str(int(color)) + ":" + str(_canonical(rows))


def _color(key):
    return int(key.split(":", 1)[0])


def _initial_edges():
    edges = {}
    for color, patterns in _PATTERNS.items():
        ids = []
        for pattern in patterns:
            rows = [[int(v) for v in row] for row in pattern.split("/")]
            ids.append(_key(color, rows))
        for i, k in enumerate(ids):
            edges[k] = [ids[(i - 1) % 7], ids[(i + 1) % 7]]
    return edges


def _tiles(obs):
    found = []
    for y in range(58):
        x = 0
        while x < 64:
            c = int(obs[y, x])
            if c in (2, 3, 5):
                x += 1
                continue
            end = x + 1
            while end < 64 and int(obs[y, end]) == c:
                end += 1
            if end - x >= 7:
                xx = x
                while xx + 7 <= end:
                    if y > 0 and int(obs[y - 1, xx]) == c:
                        xx += 7
                        continue
                    good = True
                    for j in range(7):
                        if (int(obs[y + 6, xx + j]) != c or
                                int(obs[y + j, xx]) != c or
                                int(obs[y + j, xx + 6]) != c):
                            good = False
                            break
                    if good:
                        rows = []
                        has_ink = False
                        for dy in range(1, 6):
                            row = []
                            for dx in range(1, 6):
                                v = int(obs[y + dy, xx + dx])
                                if v != c and v != 5:
                                    good = False
                                bit = int(v == 5)
                                has_ink |= bool(bit)
                                row.append(bit)
                            rows.append(row)
                        if good and has_ink:
                            found.append({
                                "x": xx,
                                "y": y,
                                "color": c,
                                "id": _key(c, rows),
                            })
                    xx += 7
            x = end
    return found


def _read_scene(obs):
    tiles = _tiles(obs)
    if not tiles:
        return None

    split = 64
    for y in range(64):
        if sum(int(obs[y, x]) == 3 for x in range(64)) >= 48:
            split = y
            break

    upper = [t for t in tiles if t["y"] < split]
    lower = [t for t in tiles if t["y"] >= split]
    ys = sorted(set(t["y"] for t in lower))
    if len(ys) < 2:
        return None

    text = sorted(
        [t for t in lower if t["y"] == ys[0]],
        key=lambda t: t["x"],
    )
    output = sorted(
        [t for t in lower if t["y"] == ys[1]],
        key=lambda t: t["x"],
    )

    groups = []
    rules = []
    for y in sorted(set(t["y"] for t in upper)):
        row = sorted(
            [t for t in upper if t["y"] == y],
            key=lambda t: t["x"],
        )
        chunks = []
        for tile in row:
            if (chunks and tile["x"] == chunks[-1][-1]["x"] + 7
                    and tile["color"] == chunks[-1][-1]["color"]):
                chunks[-1].append(tile)
            else:
                chunks.append([tile])
        for i in range(0, len(chunks) - 1, 2):
            a, b = chunks[i], chunks[i + 1]
            ia = len(groups)
            groups.extend([a, b])
            rules.append((ia, ia + 1))

    if not rules or not text or not output:
        return None

    def marked(tile):
        x = tile["x"] + 1
        y = tile["y"]
        top = y - 3
        bottom = y + 9
        if top < 0 or bottom >= 64 or x + 5 > 64:
            return False
        return (
            all(int(obs[top, x + j]) == 0 for j in range(5))
            and all(int(obs[bottom, x + j]) == 0 for j in range(5))
        )

    selected = None
    mode = None
    for i, group in enumerate(groups):
        if marked(group[0]):
            selected, mode = i, "rules"
            break
    if mode is None:
        for i, tile in enumerate(output):
            if marked(tile):
                selected, mode = i, "output"
                break
    if mode is None:
        return None

    return {
        "groups": groups,
        "rules": rules,
        "text": text,
        "output": output,
        "mode": mode,
        "selected": selected,
    }


def _learn(memory, scene):
    previous = memory.pop("previous", None)
    if not previous:
        return

    lookup = {}
    for group in scene["groups"]:
        for tile in group:
            lookup[(tile["x"], tile["y"])] = tile["id"]
    for tile in scene["output"]:
        lookup[(tile["x"], tile["y"])] = tile["id"]

    edges = memory["edges"]
    direction = previous["direction"]
    for x, y, old in previous["items"]:
        new = lookup.get((x, y))
        if new is None or new == old:
            continue
        edges.setdefault(old, [None, None])[direction] = new
        edges.setdefault(new, [None, None])[1 - direction] = old


def _path(edges, start, goal):
    if start == goal:
        return []
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        current, path = queue.popleft()
        if len(path) >= 7:
            continue
        for direction in (0, 1):
            nxt = edges.get(current, [None, None])[direction]
            if nxt is None or nxt in seen:
                continue
            p = path + [direction]
            if nxt == goal:
                return p
            seen.add(nxt)
            queue.append((nxt, p))
    return None


def _shift(edges, value, amount):
    for _ in range(amount % 7):
        value = edges.get(value, [None, None])[1]
        if value is None:
            return None
    return value


def _cycle_known(edges, value):
    seen = set()
    start = value
    for _ in range(7):
        if value is None or value in seen:
            return False
        seen.add(value)
        value = edges.get(value, [None, None])[1]
    return value == start


def _move(selected, target, count):
    right = (target - selected) % count
    left = (selected - target) % count
    return "ACTION4" if right <= left else "ACTION3"


def _emit_rotation(memory, scene, direction):
    if scene["mode"] == "rules":
        items = scene["groups"][scene["selected"]]
    else:
        items = [scene["output"][scene["selected"]]]
    memory["previous"] = {
        "direction": direction,
        "items": [[t["x"], t["y"], t["id"]] for t in items],
    }
    memory["last_rotation"] = True
    return ("ACTION1" if direction == 0 else "ACTION2", None, None)


def _normal_candidates(scene, tried):
    groups = scene["groups"]
    rules = [
        ([t["id"] for t in groups[a]], [t["id"] for t in groups[b]])
        for a, b in scene["rules"]
    ]
    text = [t["id"] for t in scene["text"]]
    output = scene["output"]

    def fits(seq, pos):
        return (
            pos + len(seq) <= len(output)
            and all(
                _color(v) == output[pos + i]["color"]
                for i, v in enumerate(seq)
            )
        )

    def signature(goal):
        return "|".join(v if v is not None else "_" for v in goal)

    result = []

    def add(seq):
        if len(seq) > len(output):
            return
        goal = seq + [None] * (len(output) - len(seq))
        sig = signature(goal)
        if sig not in tried and all(signature(g) != sig for g in result):
            result.append(goal)

    # Direct translation and recursive, symbol-by-symbol translation.
    for tree in (False, True):
        pos = 0
        seq = []
        valid = True
        while pos < len(text):
            chosen = None
            for lhs, rhs in rules:
                if text[pos:pos + len(lhs)] != lhs:
                    continue
                translated = rhs
                if tree:
                    translated = []
                    for symbol in rhs:
                        match = next(
                            (r for l, r in rules if l and l[0] == symbol),
                            None,
                        )
                        if match is None:
                            translated = None
                            break
                        translated.extend(match)
                    if translated is None:
                        continue
                chosen = (lhs, translated)
                break
            if chosen is None or not fits(chosen[1], len(seq)):
                valid = False
                break
            pos += len(chosen[0])
            seq.extend(chosen[1])
        if valid:
            add(seq)

    if result:
        return result

    # Enumerate plausible compositions if direct translations are exhausted.
    nodes = [0]

    def dfs(pos, seq):
        nodes[0] += 1
        if nodes[0] > 10000 or len(result) >= 64:
            return
        if pos >= len(text):
            add(seq)
            return
        for lhs, rhs in rules:
            if text[pos:pos + len(lhs)] != lhs:
                continue
            extensions = [([], [])] + rules
            for extra_l, extra_r in extensions:
                consumed = len(lhs) + len(extra_l)
                if pos + consumed > len(text):
                    continue
                middle = rhs + extra_r
                for lhs2, rhs2 in rules:
                    for extra_l2, extra_r2 in extensions:
                        second_lhs = lhs2 + extra_l2
                        if not all(
                            a == b for a, b in zip(middle, second_lhs)
                        ):
                            continue
                        final = rhs2 + extra_r2
                        if fits(final, len(seq)):
                            dfs(pos + consumed, seq + final)

    dfs(0, [])
    return result


def _rule_candidates(scene, edges, tried):
    groups = [[t["id"] for t in g] for g in scene["groups"]]
    rules = scene["rules"]
    text = [t["id"] for t in scene["text"]]
    output = [t["id"] for t in scene["output"]]

    shifted = []
    for group in groups:
        versions = []
        for q in range(7):
            versions.append([_shift(edges, value, q) for value in group])
        shifted.append(versions)

    def options(group_index, target):
        return [
            q for q in range(7)
            if shifted[group_index][q] == target
        ]

    def verify(assign):
        p = q = 0
        while p < len(text):
            matched = None
            for a, b in rules:
                lhs = shifted[a][assign.get(a, 0)]
                if text[p:p + len(lhs)] == lhs:
                    matched = (a, b, lhs)
                    break
            if matched is None:
                return None
            a, b, lhs = matched
            rhs = shifted[b][assign.get(b, 0)]
            if output[q:q + len(rhs)] != rhs:
                return (a, b)
            p += len(lhs)
            q += len(rhs)
        return True

    answers = []
    seen = set()
    nodes = [0]

    def save(assign):
        goals = [
            shifted[i][assign.get(i, 0)]
            for i in range(len(groups))
        ]
        sig = "|".join(",".join(g) for g in goals)
        if sig not in tried and sig not in seen:
            seen.add(sig)
            cost = sum(min(v, 7 - v) for v in assign.values())
            answers.append((cost, goals))

    def repair(assign, depth):
        if depth > 3:
            return
        check = verify(assign)
        if check is True:
            save(assign)
            return
        if check is None:
            return
        a, b = check
        for index in (a, b):
            if index in assign:
                continue
            for q in range(1, 7):
                if None in shifted[index][q]:
                    continue
                new = dict(assign)
                new[index] = q
                repair(new, depth + 1)

    def dfs(p, q, assign):
        nodes[0] += 1
        if nodes[0] > 50000 or len(answers) >= 100:
            return
        if p == len(text):
            repair(assign, 0)
            return
        for a, b in rules:
            la, lb = len(groups[a]), len(groups[b])
            if p + la > len(text) or q + lb > len(output):
                continue
            left = options(a, text[p:p + la])
            right = options(b, output[q:q + lb])
            for x in left:
                if a in assign and assign[a] != x:
                    continue
                for y in right:
                    if b in assign and assign[b] != y:
                        continue
                    new = dict(assign)
                    new[a] = x
                    new[b] = y
                    dfs(p + la, q + lb, new)

    dfs(0, 0, {})
    answers.sort(key=lambda item: item[0])
    return [goal for _, goal in answers]


def _goal_signature(mode, goal):
    if mode == "rules":
        return "|".join(",".join(group) for group in goal)
    return "|".join(v if v is not None else "_" for v in goal)


def _candidate_cost(scene, edges, goal):
    selected = scene["selected"]
    if scene["mode"] == "output":
        current = [[t["id"]] for t in scene["output"]]
        targets = [
            [v] if v is not None else [t["id"]]
            for t, v in zip(scene["output"], goal)
        ]
    else:
        current = [[t["id"] for t in g] for g in scene["groups"]]
        targets = goal

    n = len(current)
    cost = 0
    locations = []
    for i, (a, b) in enumerate(zip(current, targets)):
        if a == b:
            continue
        locations.append(i)
        j = next(k for k in range(len(a)) if a[k] != b[k])
        path = _path(edges, a[j], b[j])
        cost += len(path) if path is not None else 3
    if locations:
        cost += min(
            min((i - selected) % n, (selected - i) % n)
            for i in locations
        )
    return cost


def act(obs, available, memory):
    if "edges" not in memory:
        memory["edges"] = _initial_edges()

    scene = _read_scene(obs)
    if scene is None:
        return ("ACTION1", None, None)

    _learn(memory, scene)
    edges = memory["edges"]
    mode = scene["mode"]
    selected = scene["selected"]
    memory.setdefault("tried", [])

    # Rule-editing levels require full cycles for constraint planning.
    if mode == "rules":
        probe = memory.get("probe")
        if probe:
            index, remaining, color = probe
            if selected != index:
                memory["last_rotation"] = False
                return (
                    _move(selected, index, len(scene["groups"])),
                    None,
                    None,
                )
            if remaining > 0:
                probe[1] -= 1
                if probe[1] == 0:
                    memory.setdefault("probed", []).append(color)
                    memory.pop("probe", None)
                return _emit_rotation(memory, scene, 1)
            memory.pop("probe", None)

        probed = memory.setdefault("probed", [])
        for i, group in enumerate(scene["groups"]):
            color = group[0]["color"]
            if color not in probed and any(
                not _cycle_known(edges, t["id"]) for t in group
            ):
                memory["probe"] = [i, 7, color]
                if selected != i:
                    memory["last_rotation"] = False
                    return (
                        _move(selected, i, len(scene["groups"])),
                        None,
                        None,
                    )
                memory["probe"][1] -= 1
                return _emit_rotation(memory, scene, 1)

    plan = memory.get("plan")
    if plan is not None and memory.get("plan_mode") != mode:
        plan = None
        memory.pop("plan", None)

    if plan is None:
        tried = set(memory["tried"])
        if mode == "output":
            candidates = _normal_candidates(scene, tried)
        else:
            candidates = _rule_candidates(scene, edges, tried)
        if not candidates:
            return _emit_rotation(memory, scene, 1)
        plan = min(
            candidates,
            key=lambda g: _candidate_cost(scene, edges, g),
        )
        memory["plan"] = plan
        memory["plan_mode"] = mode

    if mode == "output":
        current = [[t["id"]] for t in scene["output"]]
        targets = [
            [v] if v is not None else [t["id"]]
            for t, v in zip(scene["output"], plan)
        ]
    else:
        current = [[t["id"] for t in g] for g in scene["groups"]]
        targets = plan

    differences = [
        i for i, (a, b) in enumerate(zip(current, targets))
        if a != b
    ]

    if not differences:
        if memory.get("last_rotation"):
            memory["tried"].append(_goal_signature(mode, plan))
            memory.pop("plan", None)
            memory["last_rotation"] = False
            return act(obs, available, memory)
        return _emit_rotation(memory, scene, 0)

    count = len(current)
    target = min(
        differences,
        key=lambda i: (
            min((i - selected) % count, (selected - i) % count),
            i,
        ),
    )
    if target != selected:
        memory["last_rotation"] = False
        return (_move(selected, target, count), None, None)

    a, b = current[target], targets[target]
    j = next(i for i in range(len(a)) if a[i] != b[i])
    path = _path(edges, a[j], b[j])
    direction = path[0] if path else 1
    return _emit_rotation(memory, scene, direction)

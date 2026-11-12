import numpy as np
import itertools
import heapq
import time


_FONTS = {
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


def _canon(a):
    a = np.asarray(a, dtype=np.uint8)
    results = []
    for _ in range(4):
        value = 0
        for bit in a.flat:
            value = (value << 1) | int(bit)
        results.append(value)
        a = np.rot90(a)
    return min(results)


def _known_fonts():
    maps, inverse = {}, {}
    for color, glyphs in _FONTS.items():
        signatures = [
            _canon([[int(c) for c in row] for row in glyph])
            for glyph in glyphs
        ]
        maps[color] = {s: i for i, s in enumerate(signatures)}
        inverse[color] = {i: s for i, s in enumerate(signatures)}
    return maps, inverse


def _parse(obs):
    tiles = []
    excluded = {0, 1, 2, 3, 4, 5}
    for y in range(58):
        row = obs[y]
        x = 0
        while x < 64:
            color = int(row[x])
            end = x + 1
            while end < 64 and int(row[end]) == color:
                end += 1
            length = end - x
            if color not in excluded and length >= 7 and length % 7 == 0:
                for xx in range(x, end, 7):
                    block = obs[y:y + 7, xx:xx + 7]
                    if block.shape != (7, 7):
                        continue
                    if not (
                        np.all(block[0] == color)
                        and np.all(block[6] == color)
                        and np.all(block[:, 0] == color)
                        and np.all(block[:, 6] == color)
                    ):
                        continue
                    inner = block[1:6, 1:6]
                    if not np.all((inner == color) | (inner == 5)):
                        continue
                    if not np.any(inner == 5):
                        continue
                    tiles.append((xx + 1, y + 1, (color, _canon(inner == 5))))
            x = end
    tiles = sorted(set(tiles), key=lambda t: (t[1], t[0]))
    if not tiles:
        return None

    boundary = None
    for y in range(1, 63):
        if np.count_nonzero(obs[y] == 3) >= 60:
            boundary = y
            break
    if boundary is None:
        return None

    upper = [t for t in tiles if t[1] < boundary]
    lower = [t for t in tiles if t[1] > boundary]
    if not upper or not lower:
        return None
    source_y = min(t[1] for t in lower)
    source = [t for t in lower if t[1] == source_y]
    target = [t for t in lower if t[1] != source_y]
    if not target:
        return None

    chains = []
    for tile in upper:
        if (
            chains
            and tile[1] == chains[-1][-1][1]
            and tile[0] == chains[-1][-1][0] + 7
        ):
            chains[-1].append(tile)
        else:
            chains.append([tile])
    if len(chains) % 2:
        return None

    rules = []
    for i in range(0, len(chains), 2):
        if chains[i][0][1] != chains[i + 1][0][1]:
            return None
        rules.append((
            tuple(t[2] for t in chains[i]),
            tuple(t[2] for t in chains[i + 1]),
        ))

    def bracket(group):
        x, y, _ = group[0]
        width = 5 + 7 * (len(group) - 1)
        yy = y - 4
        return (
            yy >= 0
            and x + width <= 64
            and np.all(obs[yy, x:x + width] == 0)
            and obs[yy + 1, x] == 0
            and obs[yy + 1, x + width - 1] == 0
        )

    alter = False
    selected = None
    for i, group in enumerate(chains):
        if bracket(group):
            alter, selected = True, i
            break
    if alter:
        editable = chains
    else:
        editable = [[t] for t in target]
        for i, group in enumerate(editable):
            if bracket(group):
                selected = i
                break
    if selected is None:
        selected = 0

    return {
        "tiles": tiles,
        "rules": rules,
        "source": tuple(t[2] for t in source),
        "target": tuple(t[2] for t in target),
        "current": tuple(tuple(t[2] for t in g) for g in editable),
        "selected": selected,
        "alter": alter,
        "key": tuple((x, y, s[0]) for x, y, s in tiles),
    }


def _learn_maps(state, scene):
    maps, inverse = _known_fonts()
    edges = state.setdefault("edges", {})
    previous = state.get("previous", {})
    action = state.get("last_action")
    if action in ("ACTION1", "ACTION2"):
        for x, y, symbol in scene["tiles"]:
            old = previous.get((x, y))
            if old is None or old[0] != symbol[0] or old == symbol:
                continue
            color = symbol[0]
            if color in maps:
                continue
            a, b = old[1], symbol[1]
            if action == "ACTION1":
                a, b = b, a
            edges.setdefault(color, {})[a] = b

    for color, successor in edges.items():
        if color in maps or not successor:
            continue
        graph = {}
        for a, b in successor.items():
            graph.setdefault(a, []).append((b, 1))
            graph.setdefault(b, []).append((a, -1))
        seen = set()
        components = []
        for root in graph:
            if root in seen:
                continue
            labels = {root: 0}
            stack = [root]
            seen.add(root)
            consistent = True
            while stack:
                a = stack.pop()
                for b, delta in graph[a]:
                    value = (labels[a] + delta) % 7
                    if b in labels:
                        if labels[b] != value:
                            consistent = False
                    else:
                        labels[b] = value
                        seen.add(b)
                        stack.append(b)
            if consistent:
                components.append(labels)
        if components:
            labels = max(components, key=len)
            maps[color] = labels
            inverse[color] = {v: s for s, v in labels.items()}
    return maps, inverse


def _link_maps(n):
    if n > 8:
        yield tuple([-1] * n)
        for start in (0, 1):
            result = [-1] * n
            for i in range(start, n - 1, 2):
                result[i], result[i + 1] = i + 1, -2
            yield tuple(result)
        return

    result = [-1] * n

    def recurse(left):
        if not left:
            yield tuple(result)
            return
        a = left[0]
        rest = left[1:]
        result[a] = -1
        yield from recurse(rest)
        for k, b in enumerate(rest):
            remaining = rest[:k] + rest[k + 1:]
            result[a], result[b] = b, -2
            yield from recurse(remaining)
            result[a], result[b] = -2, a
            yield from recurse(remaining)
            result[a], result[b] = -1, -1

    yield from recurse(list(range(n)))


def _match(seq, pos, pattern):
    return (
        pos + len(pattern) <= len(seq)
        and tuple(seq[pos:pos + len(pattern)]) == tuple(pattern)
    )


def _translate(rules, source, mode, links=None):
    n = len(rules)
    if links is None:
        links = tuple([-1] * n)
    merged = []
    for i, (left, right) in enumerate(rules):
        if links[i] >= 0:
            ll, rr = rules[links[i]]
            merged.append((left + ll, right + rr))
        else:
            merged.append((left, right))

    pos, penalty, iterations = 0, 0, 0
    output = []
    while pos < len(source):
        iterations += 1
        if iterations > len(source) + 2:
            return None
        found = False
        for i, (base_left, base_right) in enumerate(rules):
            if not _match(source, pos, base_left):
                continue
            left, right = base_left, base_right
            if mode == "tree":
                translated = []
                valid = True
                for symbol in right:
                    for inner_left, inner_right in rules:
                        if inner_left and inner_left[0] == symbol:
                            translated.extend(inner_right)
                            break
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                right = tuple(translated)
            elif mode == "double":
                if links[i] == -2:
                    continue
                left, intermediate = merged[i]
                for j in range(n):
                    if links[j] == -2:
                        continue
                    second_left, second_right = merged[j]
                    if all(a == b for a, b in zip(intermediate, second_left)):
                        right = second_right
                        break
                else:
                    continue
                if not _match(source, pos, left):
                    penalty += 1
            output.extend(right)
            pos += len(left)
            found = True
            break
        if not found:
            return None
    return tuple(output), penalty


def _navigation(changed, current, n):
    if not changed:
        return 0, 1
    vertices = set(changed)
    vertices.add(current)
    best = None
    for cut in range(n):
        positions = [(v - cut) % n for v in vertices]
        lo, hi = min(positions), max(positions)
        p = (current - cut) % n
        dl, dr = p - lo, hi - p
        cost = hi - lo + min(dl, dr)
        if p == lo:
            direction = 1
        elif p == hi:
            direction = -1
        else:
            direction = -1 if dl <= dr else 1
        candidate = (cost, direction)
        if best is None or candidate < best:
            best = candidate
    return best


def _delta(current, desired, maps):
    if current == desired:
        return 0
    differences = []
    for a, b in zip(current, desired):
        if a[0] != b[0]:
            return None
        font = maps.get(a[0], {})
        if a[1] not in font or b[1] not in font:
            return 1
        differences.append((font[b[1]] - font[a[1]]) % 7)
    if differences and len(set(differences)) == 1:
        return differences[0]
    return None


def _goal_cost(current, goal, selected, maps):
    changed, edit_cost = [], 0
    for i, (a, b) in enumerate(zip(current, goal)):
        if a == b:
            continue
        d = _delta(a, b, maps)
        if d is None:
            return float("inf")
        changed.append(i)
        edit_cost += min(d, 7 - d) if d else 1
    nav, _ = _navigation(changed, selected, len(current))
    return edit_cost + nav


def _normal_goal(scene, state, maps, deadline):
    rules = scene["rules"]
    source, target = scene["source"], scene["target"]
    current = scene["current"]
    banned = state.setdefault("banned", set())
    candidates = {}

    def add(mode, links=None):
        result = _translate(rules, source, mode, links)
        if result is None:
            return
        output, penalty = result
        if len(output) > len(target):
            return
        if any(a[0] != b[0] for a, b in zip(output, target)):
            return
        desired = output + target[len(output):]
        goal = tuple((symbol,) for symbol in desired)
        if goal in banned:
            return
        cost = _goal_cost(current, goal, scene["selected"], maps)
        if not np.isfinite(cost):
            return
        cost += 18 * (len(target) - len(output)) + 4 * penalty
        cost += {"direct": 0, "tree": 0.1, "double": 0.2}[mode]
        if goal not in candidates or cost < candidates[goal][0]:
            candidates[goal] = (cost, mode)

    add("direct")
    add("tree")
    for k, links in enumerate(_link_maps(len(rules))):
        if k % 32 == 0 and time.monotonic() > deadline:
            break
        add("double", links)
    if not candidates:
        return None
    goal = min(candidates, key=lambda g: candidates[g][0])
    return goal, candidates[goal][1]


def _find(parent, weight, x):
    value = 0
    while parent[x] != x:
        value = (value + weight[x]) % 7
        x = parent[x]
    return x, value


def _unify(parent, weight, x, y, difference):
    rx, px = _find(parent, weight, x)
    ry, py = _find(parent, weight, y)
    if rx == ry:
        return (px - py - difference) % 7 == 0
    parent[rx] = ry
    weight[rx] = (difference - px + py) % 7
    return True


def _equate(parent, weight, a, b):
    if a[0] != b[0]:
        return False
    return _unify(
        parent, weight, a[2] + 1, b[2] + 1, (b[1] - a[1]) % 7
    )


def _assignments(parent, weight, count, limit=48):
    root0, potential0 = _find(parent, weight, 0)
    fixed_value = (-potential0) % 7
    components, variable_info = {}, []
    for v in range(count):
        root, potential = _find(parent, weight, v + 1)
        variable_info.append((root, potential))
        if root != root0:
            components.setdefault(root, []).append(potential)
    roots = list(components)
    options = []
    for root in roots:
        ranked = []
        for value in range(7):
            cost = sum(
                min((value + p) % 7, (-(value + p)) % 7)
                for p in components[root]
            )
            ranked.append((cost, value))
        options.append(sorted(ranked))

    def construct(indices):
        values = {root0: fixed_value}
        for root, choices, index in zip(roots, options, indices):
            values[root] = choices[index][1]
        return tuple(
            (values[root] + potential) % 7
            for root, potential in variable_info
        )

    if not roots:
        yield construct(())
        return
    start = tuple([0] * len(roots))
    heap = [(sum(o[0][0] for o in options), start)]
    seen = {start}
    for _ in range(limit):
        if not heap:
            break
        cost, indices = heapq.heappop(heap)
        yield construct(indices)
        for i in range(len(roots)):
            if indices[i] == 6:
                continue
            nxt = list(indices)
            nxt[i] += 1
            nxt = tuple(nxt)
            if nxt in seen:
                continue
            seen.add(nxt)
            new_cost = (
                cost - options[i][indices[i]][0] + options[i][nxt[i]][0]
            )
            heapq.heappush(heap, (new_cost, nxt))


def _symbolic_options(rules, mode, links):
    n = len(rules)
    merged = []
    for i, (left, right) in enumerate(rules):
        if links[i] >= 0:
            ll, rr = rules[links[i]]
            merged.append((left + ll, right + rr))
        else:
            merged.append((left, right))

    options = []
    if mode == "direct":
        return [(left, len(left), right, ()) for left, right in rules]

    if mode == "double":
        active = [i for i in range(n) if links[i] != -2]
        for i in active:
            left, middle = merged[i]
            for j in active:
                second_left, output = merged[j]
                pairs = tuple(zip(middle, second_left))
                if any(a[0] != b[0] for a, b in pairs):
                    continue
                options.append((rules[i][0], len(left), output, pairs))
        return options

    for left, right in rules:
        choices = []
        for symbol in right:
            possible = [
                j for j in range(n)
                if rules[j][0] and rules[j][0][0][0] == symbol[0]
            ]
            if not possible:
                break
            choices.append(possible)
        else:
            for selection in itertools.product(*choices):
                output = ()
                equations = []
                for symbol, j in zip(right, selection):
                    equations.append((symbol, rules[j][0][0]))
                    output += rules[j][1]
                options.append((left, len(left), output, tuple(equations)))
    return options


def _alter_goal(scene, state, maps, inverse, deadline):
    raw_rules = scene["rules"]
    count = 2 * len(raw_rules)
    rules = []
    for i, (left, right) in enumerate(raw_rules):
        rules.append((
            tuple((c, maps[c][s], 2 * i) for c, s in left),
            tuple((c, maps[c][s], 2 * i + 1) for c, s in right),
        ))
    source = tuple((c, maps[c][s], -1) for c, s in scene["source"])
    target = tuple((c, maps[c][s], -1) for c, s in scene["target"])
    source_plain = tuple((c, d) for c, d, _ in source)
    target_plain = tuple((c, d) for c, d, _ in target)
    banned = state.setdefault("banned", set())
    failed_modes = state.setdefault("failed_modes", {})
    best = [float("inf"), None, None]
    seen_goals = set()

    def consider(parent, weight, mode, links):
        for offsets in _assignments(parent, weight, count):
            if time.monotonic() > deadline:
                return
            shifted = []
            for left, right in rules:
                shifted.append((
                    tuple((c, (d + offsets[v]) % 7) for c, d, v in left),
                    tuple((c, (d + offsets[v]) % 7) for c, d, v in right),
                ))
            result = _translate(shifted, source_plain, mode, links)
            if result is None:
                continue
            output, penalty = result
            if len(output) > len(target_plain):
                continue
            if output != target_plain[:len(output)]:
                continue
            goal = tuple(
                tuple(
                    (color, inverse[color][(maps[color][sig] + offsets[i]) % 7])
                    for color, sig in group
                )
                for i, group in enumerate(scene["current"])
            )
            if goal in banned or goal in seen_goals:
                continue
            seen_goals.add(goal)
            cost = _goal_cost(
                scene["current"], goal, scene["selected"], maps
            )
            cost += 18 * (len(target_plain) - len(output)) + 4 * penalty
            cost += 25 * failed_modes.get(mode, 0)
            cost += {"direct": 0, "tree": 0.2, "double": 0.4}[mode]
            if cost < best[0]:
                best[:] = [cost, goal, mode]

    def search_mode(mode, links):
        options = _symbolic_options(rules, mode, links)
        if not options:
            return
        nodes, leaves, memo = [0], [0], set()

        def visit(i, j, parent, weight):
            nodes[0] += 1
            if nodes[0] > 16000 or leaves[0] > 500:
                return
            if nodes[0] % 64 == 0 and time.monotonic() > deadline:
                return
            if i >= len(source):
                leaves[0] += 1
                consider(parent, weight, mode, links)
                return
            signature = tuple(
                _find(parent, weight, k) for k in range(count + 1)
            )
            key = (i, j, signature)
            if key in memo:
                return
            memo.add(key)

            for left, consumed, output, equations in options:
                if i + len(left) > len(source) or j + len(output) > len(target):
                    continue
                if any(a[0] != b[0] for a, b in zip(left, source[i:])):
                    continue
                if any(a[0] != b[0] for a, b in zip(output, target[j:])):
                    continue
                p, w = parent.copy(), weight.copy()
                valid = True
                pairs = itertools.chain(
                    equations,
                    zip(left, source[i:i + len(left)]),
                    zip(output, target[j:j + len(output)]),
                )
                for a, b in pairs:
                    if not _equate(p, w, a, b):
                        valid = False
                        break
                if valid:
                    visit(i + consumed, j + len(output), p, w)
                if time.monotonic() > deadline:
                    return

        visit(0, 0, list(range(count + 1)), [0] * (count + 1))

    n = len(rules)
    empty_links = tuple([-1] * n)
    order = ["direct", "tree", "double"]
    modes = sorted(order, key=lambda m: (failed_modes.get(m, 0), order.index(m)))
    for mode in modes:
        if time.monotonic() > deadline:
            break
        search_mode(mode, empty_links)
        if best[1] is not None and best[2] == "direct" and not failed_modes:
            break
    if best[1] is None or failed_modes.get("double", 0):
        for k, links in enumerate(_link_maps(n)):
            if links == empty_links:
                continue
            if time.monotonic() > deadline:
                break
            search_mode("double", links)
            if best[1] is not None and k > 24:
                break
    if best[1] is None:
        return None
    return best[1], best[2]


def act(obs, available, memory):
    deadline = time.monotonic() + 8.2
    allowed = set(available)
    scene = _parse(np.asarray(obs))

    def fallback():
        for action in ("ACTION4", "ACTION2", "ACTION3", "ACTION1", "RESET"):
            if action in allowed:
                return action, None, None
        return "ACTION4", None, None

    if scene is None:
        return fallback()

    state = memory.setdefault("_tr87", {})
    if state.get("layout") != scene["key"]:
        state.clear()
        state["layout"] = scene["key"]
    maps, inverse = _learn_maps(state, scene)

    def emit(action):
        if action not in allowed:
            return fallback()
        state["previous"] = {
            (x, y): symbol for x, y, symbol in scene["tiles"]
        }
        state["last_action"] = action
        return action, None, None

    current = scene["current"]
    selected = scene["selected"]
    n = len(current)
    old_goal = state.get("goal")

    if old_goal is not None and old_goal == current:
        if state.get("last_action") in ("ACTION1", "ACTION2"):
            state.setdefault("banned", set()).add(old_goal)
            if scene["alter"]:
                mode = state.get("goal_mode", "direct")
                failed = state.setdefault("failed_modes", {})
                failed[mode] = failed.get(mode, 0) + 1
            state.pop("goal", None)
            old_goal = None

    if scene["alter"]:
        all_symbols = (
            list(scene["source"])
            + list(scene["target"])
            + [s for g in current for s in g]
        )
        needed = []
        for color, sig in all_symbols:
            if (
                sig not in maps.get(color, {})
                or len(inverse.get(color, {})) < 7
            ):
                if color not in needed:
                    needed.append(color)
        if needed:
            color = needed[0]
            groups = [
                i for i, group in enumerate(current)
                if any(s[0] == color for s in group)
            ]
            if groups:
                destination = min(
                    groups,
                    key=lambda i: min((i - selected) % n, (selected - i) % n)
                )
                if destination == selected:
                    return emit("ACTION2")
                clockwise = (destination - selected) % n
                anticlockwise = (selected - destination) % n
                return emit(
                    "ACTION4" if clockwise <= anticlockwise else "ACTION3"
                )
            return emit("ACTION2")

    if old_goal is None:
        if scene["alter"]:
            result = _alter_goal(scene, state, maps, inverse, deadline)
        else:
            result = _normal_goal(scene, state, maps, deadline)
        if result is None:
            return emit("ACTION2")
        goal, mode = result
        state["goal"] = goal
        state["goal_mode"] = mode
    else:
        goal = old_goal

    changed = [i for i in range(n) if current[i] != goal[i]]
    if not changed:
        return emit("ACTION2")
    if selected in changed:
        d = _delta(current[selected], goal[selected], maps)
        if d is None or d == 0:
            d = 1
        return emit("ACTION2" if d <= 3 else "ACTION1")
    _, direction = _navigation(changed, selected, n)
    return emit("ACTION4" if direction > 0 else "ACTION3")

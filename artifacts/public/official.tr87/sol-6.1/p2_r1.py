import numpy as np
from itertools import product


_PATTERNS = {
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


def _canonical(a):
    best = None
    for _ in range(4):
        v = 0
        for b in a.ravel():
            v = (v << 1) | int(b)
        best = v if best is None else min(best, v)
        a = np.rot90(a)
    return best


def _initialize(memory):
    edges = {}
    for color, patterns in _PATTERNS.items():
        cycle = [
            _canonical(np.array([[c == "1" for c in row] for row in p]))
            for p in patterns
        ]
        edges[color] = {
            cycle[i]: cycle[(i + 1) % 7] for i in range(7)
        }
    memory["_edges"] = edges
    memory["_failed"] = set()


def _cycles(edges):
    result = {}
    for color, graph in edges.items():
        for start in sorted(graph):
            path = []
            cur = start
            for _ in range(7):
                if cur in path or cur not in graph:
                    break
                path.append(cur)
                cur = graph[cur]
            if len(path) == 7 and cur == start:
                result[color] = path
                break
    return result


def _read_board(obs):
    tiles = []
    occupied = np.zeros((64, 64), dtype=bool)
    excluded = {0, 1, 2, 3, 4, 5}

    for y in range(58):
        for x in range(58):
            color = int(obs[y, x])
            if color in excluded or occupied[y, x]:
                continue
            block = obs[y:y + 7, x:x + 7]
            if not (
                np.all(block[0] == color)
                and np.all(block[6] == color)
                and np.all(block[:, 0] == color)
                and np.all(block[:, 6] == color)
                and np.any(block[1:6, 1:6] == 5)
            ):
                continue
            if np.any(occupied[y:y + 7, x:x + 7]):
                continue
            occupied[y:y + 7, x:x + 7] = True
            tiles.append({
                "x": x + 1,
                "y": y + 1,
                "f": color,
                "m": _canonical(block[1:6, 1:6] == 5),
            })

    if not tiles:
        return None

    full_padding = [
        y for y in range(2, 62) if np.all(obs[y] == 3)
    ]
    if full_padding:
        divider = full_padding[0]
    else:
        ys = sorted({t["y"] for t in tiles})
        if len(ys) < 3:
            return None
        divider = ys[-2] - 1

    upper = [t for t in tiles if t["y"] < divider]
    lower = [t for t in tiles if t["y"] > divider]
    lower.sort(key=lambda t: (t["y"], t["x"]))
    if not lower:
        return None

    first_y = lower[0]["y"]
    inp = [t for t in lower if t["y"] == first_y]
    out = [t for t in lower if t["y"] != first_y]
    if not inp or not out:
        return None

    rules = []
    for y in sorted({t["y"] for t in upper}):
        row = sorted(
            [t for t in upper if t["y"] == y],
            key=lambda t: t["x"],
        )
        for i in range(len(row) - 1):
            a, b = row[i], row[i + 1]
            if b["x"] - a["x"] != 10:
                continue
            gap = obs[y + 2, a["x"] + 6:b["x"] - 1]
            if len(gap) != 3 or not np.all(gap == 3):
                continue
            left = i
            while left > 0 and row[left]["x"] - row[left - 1]["x"] == 7:
                left -= 1
            right = i + 1
            while (
                right + 1 < len(row)
                and row[right + 1]["x"] - row[right]["x"] == 7
            ):
                right += 1
            rules.append((row[left:i + 1], row[i + 1:right + 1]))

    if not rules:
        return None

    upper_groups = [side for rule in rules for side in rule]

    def bracket(group):
        t = group[0]
        y, x = t["y"] - 4, t["x"]
        width = 5 + 7 * (len(group) - 1)
        if y < 0 or x + width > 64:
            return False
        return (
            np.all(obs[y, x:x + width] == 0)
            and obs[y + 1, x] == 0
            and obs[y + 1, x + width - 1] == 0
        )

    selected_upper = [
        i for i, group in enumerate(upper_groups) if bracket(group)
    ]
    if selected_upper:
        alter = True
        groups = upper_groups
        selected = selected_upper[0]
    else:
        alter = False
        groups = [[t] for t in out]
        selections = [
            i for i, group in enumerate(groups) if bracket(group)
        ]
        selected = selections[0] if selections else 0

    signature = (
        tuple((t["x"], t["y"], t["f"]) for t in tiles),
        alter,
        tuple(len(g) for g in groups),
    )
    return tiles, rules, inp, out, groups, selected, alter, signature


def _navigation(n, start, required):
    required = set(required)
    if not required or required <= {start}:
        return 0, 1
    points = sorted(required | {start})
    best = None
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        gap = (b - a) % n
        length = n - gap
        s = (start - b) % n
        if s > length:
            continue
        if s <= length - s:
            cost = length + s
            direction = -1 if s else 1
        else:
            cost = length + length - s
            direction = 1 if length - s else -1
        candidate = (cost, direction)
        if best is None or candidate < best:
            best = candidate
    return best if best is not None else (n, 1)


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.w = [0] * n

    def copy(self):
        d = _DSU(0)
        d.p = self.p.copy()
        d.w = self.w.copy()
        return d

    def find(self, a):
        w = 0
        while self.p[a] != a:
            w = (w + self.w[a]) % 7
            a = self.p[a]
        return a, w

    def join(self, a, b, difference):
        ra, wa = self.find(a)
        rb, wb = self.find(b)
        difference %= 7
        if ra == rb:
            return (wa - wb) % 7 == difference
        self.p[ra] = rb
        self.w[ra] = (difference - wa + wb) % 7
        return True


def _solve(rules, inp, out, groups, selected, alter, cycles, failed):
    n = len(groups)
    phases = {
        f: {m: i for i, m in enumerate(c)}
        for f, c in cycles.items()
    }

    def ref(tile, gid=-1):
        return (
            gid, tile["f"], tile["m"],
            phases.get(tile["f"], {}).get(tile["m"]),
        )

    rr = []
    for i, (left, right) in enumerate(rules):
        rr.append((
            [ref(t, 2 * i if alter else -1) for t in left],
            [ref(t, 2 * i + 1 if alter else -1) for t in right],
        ))
    ir = [ref(t) for t in inp]
    ore = [ref(t) for t in out]
    nin, nout = len(ir), len(ore)
    current = tuple(
        tuple(t["m"] for t in group) for group in groups
    )

    def equality_possible(a, b):
        if a[1] != b[1]:
            return False
        return alter or a[2] == b[2]

    def constrain(d, a, b):
        if a[1] != b[1]:
            return False
        if a[0] < 0 and b[0] < 0:
            return a[2] == b[2]
        if a[3] is None or b[3] is None:
            return False
        ga = n if a[0] < 0 else a[0]
        gb = n if b[0] < 0 else b[0]
        return d.join(ga, gb, b[3] - a[3])

    def shifted(r, shifts):
        if r[0] < 0:
            return (r[1], r[2])
        return (
            r[1],
            cycles[r[1]][(r[3] + shifts[r[0]]) % 7],
        )

    def validate(mode, shifts, target):
        if mode == "double":
            return True
        material = [
            (
                [shifted(r, shifts) for r in left],
                [shifted(r, shifts) for r in right],
            )
            for left, right in rr
        ]
        input_names = [(r[1], r[2]) for r in ir]
        output_names = (
            [(r[1], r[2]) for r in ore]
            if alter else
            [(out[i]["f"], target[i][0]) for i in range(nout)]
        )
        ip = op = 0
        while ip < nin:
            found = False
            for left, right in material:
                if input_names[ip:ip + len(left)] != left:
                    continue
                translated = right
                if mode == "tree":
                    translated = []
                    possible = True
                    for symbol in right:
                        for l2, r2 in material:
                            if l2[0] == symbol:
                                translated.extend(r2)
                                break
                        else:
                            possible = False
                            break
                    if not possible:
                        continue
                if output_names[op:op + len(translated)] != translated:
                    return False
                ip += len(left)
                op += len(translated)
                found = True
                break
            if not found:
                return False
        return True

    best = None

    def consider(mode, shifts, output_target):
        nonlocal best
        if alter:
            target = tuple(
                tuple(
                    cycles[t["f"]][
                        (phases[t["f"]][t["m"]] + shifts[g]) % 7
                    ]
                    for t in group
                )
                for g, group in enumerate(groups)
            )
            deltas = list(shifts)
        else:
            target = tuple((m,) for m in output_target)
            deltas = []
            for t, goal in zip(out, output_target):
                p = phases.get(t["f"], {})
                if t["m"] not in p or goal not in p:
                    return
                deltas.append((p[goal] - p[t["m"]]) % 7)

        if target in failed:
            return
        if not validate(mode, shifts, target):
            return
        edit_cost = sum(min(d, 7 - d) for d in deltas)
        needed = [i for i, d in enumerate(deltas) if d]
        movement, _ = _navigation(n, selected, needed)
        # An already-correct image still needs an edit to trigger checking.
        score = edit_cost + movement + (2 if not needed else 0)
        candidate = (score, target)
        if best is None or candidate < best:
            best = candidate

    def finish_alter(mode, d):
        root_members = {}
        for g in range(n):
            root, offset = d.find(g)
            root_members.setdefault(root, []).append((g, offset))
        constant_root, constant_offset = d.find(n)
        fixed_value = (-constant_offset) % 7

        beam = [(0, {})]
        for root, members in root_members.items():
            if root == constant_root:
                choices = [fixed_value]
            else:
                ranked = []
                for value in range(7):
                    cost = sum(
                        min((value + offset) % 7,
                            7 - (value + offset) % 7)
                        for _, offset in members
                    )
                    ranked.append((cost, value))
                ranked.sort()
                minimum = ranked[0][0]
                choices = [
                    v for c, v in ranked
                    if c <= minimum + 1
                ][:3]
            expanded = []
            for old_cost, assignment in beam:
                for value in choices:
                    cost = sum(
                        min((value + offset) % 7,
                            7 - (value + offset) % 7)
                        for _, offset in members
                    )
                    a = assignment.copy()
                    a[root] = value
                    expanded.append((old_cost + cost, a))
            expanded.sort(key=lambda z: z[0])
            beam = expanded[:32]

        for _, assignment in beam:
            shifts = []
            for g in range(n):
                root, offset = d.find(g)
                shifts.append((assignment[root] + offset) % 7)
            consider(mode, shifts, None)

    def make_maps(mode):
        if mode == "base":
            return [
                (left, right, [])
                for left, right in rr
                if len(left) <= nin and len(right) <= nout
            ]

        if mode == "tree":
            maps = []
            for left, intermediate in rr:
                choices = []
                for symbol in intermediate:
                    eligible = [
                        j for j, (l2, r2) in enumerate(rr)
                        if equality_possible(symbol, l2[0])
                    ]
                    if not alter:
                        eligible = eligible[:1]
                    choices.append(eligible)
                if any(not c for c in choices):
                    continue
                for picks in product(*choices):
                    right, equations = [], []
                    for symbol, j in zip(intermediate, picks):
                        l2, r2 = rr[j]
                        right.extend(r2)
                        equations.append((symbol, l2[0]))
                    if len(left) <= nin and len(right) <= nout:
                        maps.append((left, right, equations))
                    if len(maps) >= 12000:
                        return maps
            return maps

        combined = [(l, r) for l, r in rr if len(l) <= nin]
        for i, (l1, r1) in enumerate(rr):
            for j, (l2, r2) in enumerate(rr):
                if i != j and len(l1) + len(l2) <= nin:
                    combined.append((l1 + l2, r1 + r2))
        combined = combined[:300]
        maps = []
        for left, intermediate in combined:
            for l2, right in combined:
                if len(right) > nout:
                    continue
                equations = list(zip(intermediate, l2))
                if all(equality_possible(a, b) for a, b in equations):
                    maps.append((left, right, equations))
                if len(maps) >= 20000:
                    return maps
        return maps

    for mode in ("base", "tree", "double"):
        best = None
        maps = make_maps(mode)
        # Shorter mappings are useful first and reduce branching.
        maps.sort(key=lambda z: (len(z[0]), len(z[1]), len(z[2])))
        visits = [0]
        leaves = [0]
        seen = set()

        def search(ip, op, d, target):
            if visits[0] >= 14000 or leaves[0] >= 180:
                return
            visits[0] += 1
            if ip == nin:
                leaves[0] += 1
                if alter:
                    finish_alter(mode, d)
                else:
                    consider(mode, [0] * n, target)
                return

            if alter:
                state = (ip, op, tuple(d.p), tuple(d.w))
            else:
                state = (ip, op, tuple(target))
            if state in seen:
                return
            seen.add(state)

            for left, right, equations in maps:
                if ip + len(left) > nin or op + len(right) > nout:
                    continue
                if any(
                    a[1] != b[1]
                    for a, b in zip(left, ir[ip:ip + len(left)])
                ):
                    continue
                if any(
                    a[1] != b[1]
                    for a, b in zip(right, ore[op:op + len(right)])
                ):
                    continue

                if alter:
                    nd = d.copy()
                    ok = True
                    pairs = (
                        list(zip(left, ir[ip:ip + len(left)]))
                        + list(zip(right, ore[op:op + len(right)]))
                        + equations
                    )
                    for a, b in pairs:
                        if not constrain(nd, a, b):
                            ok = False
                            break
                    if ok:
                        search(ip + len(left), op + len(right), nd, target)
                else:
                    if any(
                        a[2] != b[2]
                        for a, b in zip(left, ir[ip:ip + len(left)])
                    ):
                        continue
                    if any(a[2] != b[2] for a, b in equations):
                        continue
                    nt = target.copy()
                    nt[op:op + len(right)] = [r[2] for r in right]
                    search(ip + len(left), op + len(right), d, nt)

                if visits[0] >= 14000 or leaves[0] >= 180:
                    break

        search(0, 0, _DSU(n + 1), [t["m"] for t in out])
        if best is not None:
            return best[1]

    return None


def act(obs, available, memory):
    if "_edges" not in memory:
        _initialize(memory)

    board = _read_board(obs)
    if board is None:
        action = "ACTION2" if "ACTION2" in available else available[0]
        return action, None, None

    tiles, rules, inp, out, groups, selected, alter, signature = board
    snapshot = {
        (t["x"], t["y"]): (t["f"], t["m"]) for t in tiles
    }

    if memory.get("_signature") != signature:
        memory["_signature"] = signature
        memory["_failed"] = set()
        memory.pop("_plan", None)
        memory.pop("_pending", None)
        memory.pop("_previous", None)

    previous = memory.get("_previous")
    last_action = memory.get("_last_action")
    if previous is not None and last_action in ("ACTION1", "ACTION2"):
        for position, (family, mask) in snapshot.items():
            old = previous.get(position)
            if old is None or old[0] != family or old[1] == mask:
                continue
            graph = memory["_edges"].setdefault(family, {})
            if last_action == "ACTION2":
                graph[old[1]] = mask
            else:
                graph[mask] = old[1]

    pending = memory.pop("_pending", None)
    if pending is not None:
        # A predicted final edit did not finish the episode.
        memory["_failed"].add(pending)
        memory.pop("_plan", None)

    cycles = _cycles(memory["_edges"])
    phases = {
        f: {m: i for i, m in enumerate(cycle)}
        for f, cycle in cycles.items()
    }

    def emit(action):
        if action not in available:
            action = next(
                (a for a in ("ACTION2", "ACTION1", "ACTION4", "ACTION3")
                 if a in available),
                available[0],
            )
        memory["_previous"] = snapshot
        memory["_last_action"] = action
        return action, None, None

    # Learn cycles only for symbol families that are actually editable.
    unknown_groups = []
    for i, group in enumerate(groups):
        if any(
            t["f"] not in phases or t["m"] not in phases[t["f"]]
            for t in group
        ):
            unknown_groups.append(i)
    if unknown_groups:
        if selected in unknown_groups:
            return emit("ACTION2")
        _, direction = _navigation(len(groups), selected, unknown_groups)
        return emit("ACTION4" if direction > 0 else "ACTION3")

    target = memory.get("_plan")
    if target is None:
        target = _solve(
            rules, inp, out, groups, selected, alter,
            cycles, memory["_failed"],
        )
        if target is not None:
            memory["_plan"] = target

    if target is None:
        # Retry alternative configurations without resetting progress.
        memory["_failed"].clear()
        return emit("ACTION2")

    deltas = []
    valid_plan = len(target) == len(groups)
    if valid_plan:
        for group, desired in zip(groups, target):
            if len(group) != len(desired):
                valid_plan = False
                break
            local = []
            for tile, mask in zip(group, desired):
                p = phases.get(tile["f"], {})
                if mask not in p or tile["m"] not in p:
                    valid_plan = False
                    break
                local.append((p[mask] - p[tile["m"]]) % 7)
            if not valid_plan or len(set(local)) != 1:
                valid_plan = False
                break
            deltas.append(local[0])

    if not valid_plan:
        memory.pop("_plan", None)
        return emit("ACTION2")

    needed = [i for i, delta in enumerate(deltas) if delta]
    if not needed:
        # The simulator checks the goal only after a symbol-changing action.
        return emit("ACTION2")

    if deltas[selected]:
        delta = deltas[selected]
        action = "ACTION2" if delta <= 3 else "ACTION1"
        if sum(min(d, 7 - d) for d in deltas) == 1:
            memory["_pending"] = target
        return emit(action)

    _, direction = _navigation(len(groups), selected, needed)
    return emit("ACTION4" if direction > 0 else "ACTION3")

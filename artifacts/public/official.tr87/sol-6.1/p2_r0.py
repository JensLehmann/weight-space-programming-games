import math
import time
from itertools import product
import numpy as np


# Unrotated masks, in the order used by the cyclic editing actions.
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
    a = np.asarray(a, dtype=np.uint8)
    values = []
    for k in range(4):
        b = np.rot90(a, k).ravel()
        v = 0
        for x in b:
            v = (v << 1) | int(x)
        values.append(v)
    return min(values)


def _tables(memory):
    if "_tables" not in memory:
        tables = {}
        for color, patterns in _PATTERNS.items():
            tables[color] = [
                _canonical([[int(c) for c in row] for row in p])
                for p in patterns
            ]
        memory["_tables"] = tables

    tables = memory["_tables"]
    edges = memory.setdefault("_edges", {})
    # A newly discovered family can have an arbitrary index origin.
    families = {k[0] for k in edges}
    for color in families:
        if color in tables:
            continue
        starts = [k[1] for k in edges if k[0] == color]
        for start in starts:
            seq = []
            cur = start
            while cur not in seq and len(seq) < 8:
                seq.append(cur)
                nxt = edges.get((color, cur))
                if nxt is None:
                    break
                cur = nxt
            if len(seq) == 7 and cur == start:
                tables[color] = seq
                break
    return tables


def _read(obs):
    """Return tiles, upper rule groups, and the two lower sequences."""
    tiles = []
    occupied = np.zeros((64, 64), dtype=bool)
    excluded = {0, 1, 2, 3, 4, 5}
    for y in range(58):
        for x in range(58):
            if occupied[y, x]:
                continue
            color = int(obs[y, x])
            if color in excluded:
                continue
            if not np.all(obs[y, x:x + 7] == color):
                continue
            if not np.all(obs[y + 6, x:x + 7] == color):
                continue
            if not np.all(obs[y:y + 7, x] == color):
                continue
            if not np.all(obs[y:y + 7, x + 6] == color):
                continue
            core = obs[y + 1:y + 6, x + 1:x + 6]
            if not np.all((core == color) | (core == 5)):
                continue
            if np.count_nonzero(core == 5) < 3:
                continue
            key = (color, _canonical(core == 5))
            tiles.append((x, y, key))
            occupied[y:y + 7, x:x + 7] = True

    separators = [
        y for y in range(64)
        if np.all(obs[y] == 3)
    ]
    if not separators or not tiles:
        return None
    split = min(separators)
    upper = [t for t in tiles if t[1] < split]
    lower = [t for t in tiles if t[1] > split]
    ys = sorted({t[1] for t in lower})
    if len(ys) < 2:
        return None
    inp = sorted([t for t in lower if t[1] == ys[0]])
    out = sorted([t for t in lower if t[1] == ys[1]])

    groups = []
    for y in sorted({t[1] for t in upper}):
        row = sorted([t for t in upper if t[1] == y])
        rowgroups = []
        for t in row:
            if not rowgroups or t[0] != rowgroups[-1][-1][0] + 7:
                rowgroups.append([t])
            else:
                rowgroups[-1].append(t)
        groups.extend(rowgroups)
    if len(groups) % 2:
        return None
    rules = [(groups[i], groups[i + 1])
             for i in range(0, len(groups), 2)]
    return rules, groups, inp, out


def _cap(obs, group):
    x, y, _ = group[0]
    x += 1
    y -= 3
    width = 7 * len(group) - 2
    return (0 <= y < 64 and x + width <= 64
            and np.all(obs[y, x:x + width] == 0))


def _distance(a, b, n):
    return min((a - b) % n, (b - a) % n)


def _walk_cost(start, needed, n):
    if not needed:
        return 0
    points = sorted(set(needed) | {start})
    if len(points) == 1:
        return 0
    best = 2 * n
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        gap = (q - p) % n
        length = n - gap
        position = (start - q) % n
        if position <= length:
            best = min(best, length + min(position, length - position))
    return best


def _navigate(current, target, n):
    cw = (target - current) % n
    ccw = (current - target) % n
    return "ACTION4" if cw <= ccw else "ACTION3"


def _rotmask(mask, amount):
    amount %= 7
    if amount == 0:
        return mask
    return ((mask << amount) | (mask >> (7 - amount))) & 127


def _propagate(domains, equations):
    domains = list(domains)
    for _ in range(2 * len(domains) + 2):
        changed = False
        for u, v, d in equations:
            a = domains[u] & _rotmask(domains[v], d)
            if not a:
                return None
            b = domains[v] & _rotmask(a, -d)
            if not b:
                return None
            if a != domains[u] or b != domains[v]:
                domains[u], domains[v] = a, b
                changed = True
        if not changed:
            return domains
    return domains


def _solve(rules, controls, inp, out, altered, cursor, tables,
           failed, deadline):
    """
    Every editable group is a modulo-seven variable.
    Equality of symbols becomes a difference equation between variables.
    """
    n = len(controls)
    constant = n
    inverse = {
        color: {mask: i for i, mask in enumerate(seq)}
        for color, seq in tables.items()
    }

    def expression(tile, variable):
        color, mask = tile[2]
        return (color, inverse[color][mask], variable)

    group_ids = {}
    if altered:
        for i, group in enumerate(controls):
            group_ids[(group[0][0], group[0][1])] = i

    symbolic_rules = []
    for left, right in rules:
        gl = group_ids[(left[0][0], left[0][1])] if altered else constant
        gr = group_ids[(right[0][0], right[0][1])] if altered else constant
        symbolic_rules.append((
            tuple(expression(t, gl) for t in left),
            tuple(expression(t, gr) for t in right),
        ))
    input_expr = tuple(expression(t, constant) for t in inp)
    output_expr = tuple(
        expression(t, constant if altered else i)
        for i, t in enumerate(out)
    )

    def equal(a, b):
        if a[0] != b[0]:
            return None
        d = (b[1] - a[1]) % 7
        if a[2] == b[2]:
            return [] if d == 0 else None
        return [(a[2], b[2], d)]

    def equal_lists(a, b):
        equations = []
        for x, y in zip(a, b):
            e = equal(x, y)
            if e is None:
                return None
            equations.extend(e)
        return equations

    input_colors = tuple(t[0] for t in input_expr)
    output_colors = tuple(t[0] for t in output_expr)

    def fits_anywhere(expr, colors):
        f = tuple(t[0] for t in expr)
        return any(colors[p:p + len(f)] == f
                   for p in range(len(colors) - len(f) + 1))

    transforms = []
    seen = set()

    def add(left, right, equations, penalty):
        if not left or not right:
            return
        if len(left) > len(inp) or len(right) > len(out):
            return
        if not fits_anywhere(left, input_colors):
            return
        if not fits_anywhere(right, output_colors):
            return
        equations = tuple(sorted(set(equations)))
        key = (left, right, equations)
        if key in seen:
            return
        seen.add(key)
        transforms.append((left, right, equations, penalty))

    # Direct rules.
    for left, right in symbolic_rules:
        add(left, right, [], 0.0)

    # Joined rules model the invisible links used by double translation.
    variants = [(l, r, 0.0) for l, r in symbolic_rules]
    for i, (left, right) in enumerate(symbolic_rules):
        for j, (left2, right2) in enumerate(symbolic_rules):
            if i == j:
                continue
            if len(left) + len(left2) <= len(inp) + len(out):
                variants.append((left + left2, right + right2, 0.8))

    # Translate an entire intermediate sequence a second time.
    for left, middle, penalty in variants:
        if not fits_anywhere(left, input_colors):
            continue
        for second_left, right, penalty2 in variants:
            if time.monotonic() > deadline:
                break
            eq = equal_lists(middle, second_left)
            if eq is None:
                continue
            mismatch_penalty = 0 if len(middle) == len(second_left) else 8
            add(left, right, eq,
                0.15 + penalty + penalty2 + mismatch_penalty)

    # Tree translation: translate each intermediate token separately.
    for left, middle in symbolic_rules:
        if not fits_anywhere(left, input_colors):
            continue

        def expand(k, result, equations):
            if time.monotonic() > deadline:
                return
            if k == len(middle):
                add(left, result, equations, 0.2)
                return
            for second_left, right in symbolic_rules:
                eq = equal(middle[k], second_left[0])
                if eq is None or len(result) + len(right) > len(out):
                    continue
                expand(k + 1, result + right, equations + eq)

        expand(0, (), [])

    transforms.sort(key=lambda t: t[3])
    initial_domains = [127] * n + [1]
    prepared = []
    for left, right, equations, penalty in transforms:
        domains = _propagate(initial_domains, equations)
        if domains is not None:
            prepared.append((left, right, equations, penalty))

    current_keys = tuple(tuple(t[2] for t in g) for g in controls)
    best = None
    best_cost = float("inf")
    nodes = 0
    leaves = 0

    def consider(domains, equations, penalty, output_end):
        nonlocal best, best_cost, leaves
        leaves += 1
        adjacency = [[] for _ in range(n + 1)]
        for u, v, d in equations:
            adjacency[v].append((u, d))
            adjacency[u].append((v, (-d) % 7))

        components = []
        visited = set()
        for root in range(n + 1):
            if root in visited:
                continue
            offsets = {root: 0}
            stack = [root]
            visited.add(root)
            while stack:
                u = stack.pop()
                for v, d in adjacency[u]:
                    if v not in offsets:
                        offsets[v] = (offsets[u] + d) % 7
                        visited.add(v)
                        stack.append(v)

            choices = []
            for value in range(7):
                vals = {
                    u: (value + offset) % 7
                    for u, offset in offsets.items()
                }
                if not all(domains[u] & (1 << val)
                           for u, val in vals.items()):
                    continue
                cost = sum(min(val, 7 - val)
                           for u, val in vals.items() if u < n)
                choices.append((cost, vals))
            choices.sort(key=lambda item: item[0])
            if not choices:
                return
            # A completely unused variable should simply remain unchanged.
            if len(offsets) == 1 and root < n and not adjacency[root]:
                choices = [next(c for c in choices if c[1][root] == 0)]
            components.append(choices)

        beam = [(0, [0] * n)]
        for choices in components:
            next_beam = []
            for cost, assignment in beam:
                for extra, vals in choices:
                    a = assignment[:]
                    for u, val in vals.items():
                        if u < n:
                            a[u] = val
                    next_beam.append((cost + extra, a))
            next_beam.sort(key=lambda item: item[0])
            beam = next_beam[:48]

        for edit_cost, assignment in beam:
            goals = []
            for i, group in enumerate(controls):
                target = []
                for tile in group:
                    color, mask = tile[2]
                    index = inverse[color][mask]
                    target.append((
                        color,
                        tables[color][(index + assignment[i]) % 7],
                    ))
                goals.append(tuple(target))
            signature = tuple(goals)
            if signature in failed:
                continue
            needed = [i for i, shift in enumerate(assignment) if shift]
            movement = _walk_cost(cursor, needed, n)
            # Official layouts normally consume both strings completely.
            leftover = len(out) - output_end
            cost = (edit_cost + movement + penalty
                    + (20 + 3 * leftover if leftover else 0))
            if not needed:
                cost += 2
            if cost < best_cost:
                best_cost = cost
                best = signature

    # Index transformations by the family of their initial input token.
    by_family = {}
    for t in prepared:
        by_family.setdefault(t[0][0][0], []).append(t)

    def search(p, q, domains, equations, penalty):
        nonlocal nodes
        nodes += 1
        if (nodes > 24000 or leaves > 1800
                or time.monotonic() > deadline):
            return
        if p == len(input_expr):
            consider(domains, equations, penalty, q)
            return
        for left, right, extra, extra_penalty in by_family.get(
                input_expr[p][0], []):
            if p + len(left) > len(inp) or q + len(right) > len(out):
                continue
            e1 = equal_lists(left, input_expr[p:p + len(left)])
            if e1 is None:
                continue
            e2 = equal_lists(right, output_expr[q:q + len(right)])
            if e2 is None:
                continue
            new_eq = list(extra) + e1 + e2
            new_domains = _propagate(domains, new_eq)
            if new_domains is None:
                continue
            search(p + len(left), q + len(right), new_domains,
                   equations + new_eq, penalty + extra_penalty)

    search(0, 0, initial_domains, [], 0.0)
    return best


def act(obs, available, memory):
    obs = np.asarray(obs)
    deadline = time.monotonic() + 7.5

    def finish(action):
        if action not in available:
            usable = [a for a in available if a != "RESET"]
            action = usable[0] if usable else available[0]
        return action, None, None

    parsed = _read(obs)
    if parsed is None:
        return finish("ACTION2")

    rules, upper_groups, inp, out = parsed
    top_selected = [
        i for i, group in enumerate(upper_groups) if _cap(obs, group)
    ]
    altered = bool(top_selected)
    if not top_selected and memory.get("_altered", False):
        # Cursor recognition is normally exact; retain the mode if needed.
        altered = True
    memory["_altered"] = altered
    controls = upper_groups if altered else [[t] for t in out]
    n = len(controls)
    if not n:
        return finish("ACTION2")

    selected = [i for i, g in enumerate(controls) if _cap(obs, g)]
    cursor = selected[0] if selected else memory.get("_cursor", 0) % n
    memory["_cursor"] = cursor

    keys = tuple(tuple(t[2] for t in g) for g in controls)

    # Record observed cyclic transitions, including simultaneous group edits.
    previous = memory.pop("_last_rotation", None)
    if previous is not None:
        old_index, old_keys, direction = previous
        if old_index < n and len(old_keys) == len(keys[old_index]):
            edges = memory.setdefault("_edges", {})
            for old, new in zip(old_keys, keys[old_index]):
                if old[0] == new[0]:
                    if direction == 1:
                        edges[old] = new[1]
                    else:
                        edges[new] = old[1]

    tables = _tables(memory)
    inverse = {
        c: {mask: i for i, mask in enumerate(seq)}
        for c, seq in tables.items()
    }

    def emit(action, editing=False):
        if action == "ACTION3":
            memory["_cursor"] = (cursor - 1) % n
        elif action == "ACTION4":
            memory["_cursor"] = (cursor + 1) % n
        elif action in ("ACTION1", "ACTION2"):
            direction = -1 if action == "ACTION1" else 1
            memory["_last_rotation"] = (
                cursor, keys[cursor], direction
            )
            if editing and "_goal" in memory:
                memory["_edited"] = True
        return finish(action)

    all_tiles = inp + out + [
        t for group in upper_groups for t in group
    ]
    unknown = {
        t[2][0] for t in all_tiles
        if t[2][0] not in inverse
        or t[2][1] not in inverse[t[2][0]]
    }
    if unknown:
        candidates = [
            i for i, group in enumerate(controls)
            if any(t[2][0] in unknown for t in group)
        ]
        if candidates:
            target = min(candidates,
                         key=lambda i: (_distance(cursor, i, n), i))
            if cursor != target:
                return emit(_navigate(cursor, target, n))
            return emit("ACTION2")
        return emit("ACTION2")

    # Discard a plan only after it has actually been tested by an edit.
    failed = memory.setdefault("_failed", set())
    goal = memory.get("_goal")
    if goal is not None and len(goal) != n:
        memory.pop("_goal", None)
        goal = None
    if goal is not None and keys == goal and memory.get("_edited", False):
        failed.add(goal)
        memory.pop("_goal", None)
        memory.pop("_edited", None)
        goal = None

    if goal is None:
        goal = _solve(
            rules, controls, inp, out, altered, cursor,
            tables, failed, deadline
        )
        if goal is None:
            # Revisit interpretations if all recorded candidates were exhausted.
            if failed and time.monotonic() < deadline - 0.5:
                failed.clear()
                goal = _solve(
                    rules, controls, inp, out, altered, cursor,
                    tables, failed, deadline
                )
        if goal is None:
            return emit("ACTION2")
        memory["_goal"] = goal
        memory["_edited"] = False

    wrong = [i for i in range(n) if keys[i] != goal[i]]
    if not wrong:
        # Completion is checked only after a symbol-changing action.
        # Move away once; the normal goal-following logic will restore it.
        return emit("ACTION2", editing=True)

    target = min(
        wrong,
        key=lambda i: (_distance(cursor, i, n), i)
    )
    if target != cursor:
        return emit(_navigate(cursor, target, n))

    # All members of an editable rule group share the same cyclic offset.
    for current, wanted in zip(keys[cursor], goal[cursor]):
        if current != wanted:
            color = current[0]
            a = inverse[color][current[1]]
            b = inverse[color][wanted[1]]
            forward = (b - a) % 7
            backward = (a - b) % 7
            action = "ACTION2" if forward <= backward else "ACTION1"
            return emit(action, editing=True)

    return emit("ACTION2", editing=True)

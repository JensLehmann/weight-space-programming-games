import numpy as np
import itertools
import heapq
import time


# Glyph identities are rotation invariant; their order determines ACTION1/2.
_GLYPHS = {
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


def _canon(a):
    best = None
    for k in range(4):
        z = np.rot90(a, k).ravel()
        v = 0
        for b in z:
            v = (v << 1) | int(bool(b))
        if best is None or v < best:
            best = v
    return best


def _initial_maps():
    result = {}
    for color, patterns in _GLYPHS.items():
        result[color] = {}
        for i, text in enumerate(patterns):
            a = np.array([[ch == "1" for ch in row]
                          for row in text.split("/")], dtype=bool)
            result[color][_canon(a)] = i
    return result


def _raw(t):
    return (t["c"], t["k"])


def _parse(obs):
    tiles = []
    # A tile has a solid seven-pixel top edge and a five-pixel glyph.
    for y in range(57):
        row = obs[y]
        x = 0
        while x < 64:
            c = int(row[x])
            e = x + 1
            while e < 64 and int(row[e]) == c:
                e += 1
            if c not in (0, 2, 3, 5) and e - x >= 7:
                for xx in range(x, e - 6, 7):
                    block = obs[y:y + 7, xx:xx + 7]
                    inside = block[1:6, 1:6]
                    edge = np.concatenate((
                        block[-1, :], block[:, 0], block[:, -1]
                    ))
                    if (np.all((edge == c) | (edge == 0))
                            and np.all((inside == c) | (inside == 5))
                            and np.count_nonzero(inside == 5) >= 3):
                        tiles.append({
                            "x": xx, "y": y, "c": c,
                            "k": _canon(inside == 5)
                        })
            x = e

    tiles.sort(key=lambda t: (t["y"], t["x"]))
    ys = sorted(set(t["y"] for t in tiles))
    if len(ys) < 3:
        return None

    source = [t for t in tiles if t["y"] == ys[-2]]
    target = [t for t in tiles if t["y"] == ys[-1]]
    rules = []

    for y in ys[:-2]:
        ts = [t for t in tiles if t["y"] == y]
        groups = []
        for t in ts:
            if groups and groups[-1][-1]["x"] + 7 == t["x"]:
                groups[-1].append(t)
            else:
                groups.append([t])
        found = []
        for i in range(len(groups) - 1):
            left, right = groups[i], groups[i + 1]
            lo = left[-1]["x"] + 7
            hi = right[0]["x"]
            if hi > lo and np.any(obs[y + 3, lo:hi] == 3):
                found.append((left, right))
        if not found and len(groups) % 2 == 0:
            found = [(groups[i], groups[i + 1])
                     for i in range(0, len(groups), 2)]
        rules.extend(found)

    if not rules or not source or not target:
        return None

    flat = [s for rule in rules for s in rule]

    def bracket_score(group):
        x = group[0]["x"] + 1
        y = group[0]["y"]
        width = 5 + 7 * (len(group) - 1)
        if x + width > 64:
            return -1
        score = 0.0
        for yy in (y - 3, y + 9):
            if 0 <= yy < 63:
                score += float(np.mean(obs[yy, x:x + width] == 0))
        return score

    choices = []
    for i, t in enumerate(target):
        choices.append((bracket_score([t]), False, i))
    for i, s in enumerate(flat):
        choices.append((bracket_score(s), True, i))
    score, alter, cursor = max(choices, key=lambda z: z[0])
    if score < 0.7:
        alter, cursor = False, 0

    return {
        "tiles": tiles, "rules": rules,
        "source": source, "target": target,
        "alter": alter, "cursor": cursor,
        "slots": flat if alter else [[t] for t in target],
    }


def _distance(d):
    d %= 7
    return min(d, 7 - d)


def _ring_plan(deltas, cursor):
    """Shortest edit route on a cyclic list of selections."""
    n = len(deltas)
    needed = [i for i, d in enumerate(deltas) if d % 7]
    if not needed:
        return ["ACTION2", "ACTION1"]

    best = None
    for cut in range(n):
        s = (cursor - cut) % n
        positions = [(i - cut) % n for i in needed] + [s]
        lo, hi = min(positions), max(positions)
        first, last = ((lo, hi) if s - lo <= hi - s else (hi, lo))
        path = [s]
        u = s
        for endpoint in (first, last):
            while u != endpoint:
                u += 1 if endpoint > u else -1
                path.append(u)

        plan = []
        edited = set()
        previous = path[0]
        for j, u in enumerate(path):
            if j:
                plan.append("ACTION4" if u > previous else "ACTION3")
            index = (u + cut) % n
            if index not in edited:
                edited.add(index)
                d = deltas[index] % 7
                if d <= 3:
                    plan.extend(["ACTION2"] * d)
                else:
                    plan.extend(["ACTION1"] * (7 - d))
            previous = u
        if best is None or len(plan) < len(best):
            best = plan
    return best


def _move_to(current, target, n):
    right = (target - current) % n
    left = (current - target) % n
    return "ACTION4" if right <= left else "ACTION3"


def _translate(rules, source, mode, links=None, strict=False):
    links = links or {}
    seconds = set(links.values())

    def joined(i):
        a, b = rules[i]
        if i in links:
            aa, bb = rules[links[i]]
            return a + aa, b + bb
        return a, b

    p = 0
    result = []
    iterations = 0
    while p < len(source):
        iterations += 1
        if iterations > len(source) + 2:
            return None
        matched = False
        for i, (original_left, original_right) in enumerate(rules):
            if source[p:p + len(original_left)] != original_left:
                continue
            left, right = original_left, original_right

            if mode == "tree":
                expanded = []
                valid = True
                for token in right:
                    for aa, bb in rules:
                        if aa and aa[0] == token:
                            expanded.extend(bb)
                            break
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                right = expanded

            elif mode == "double":
                if i in seconds:
                    continue
                left, intermediate = joined(i)
                if strict and source[p:p + len(left)] != left:
                    return None
                valid = False
                for j in range(len(rules)):
                    if j in seconds:
                        continue
                    aa, bb = joined(j)
                    if all(a == b for a, b in zip(intermediate, aa)):
                        if strict and len(intermediate) != len(aa):
                            return None
                        right = bb
                        valid = True
                        break
                if not valid:
                    continue

            result.extend(right)
            p += len(left)
            matched = True
            break
        if not matched:
            return None
    return result


def _model_key(mode, links):
    return (mode, tuple(sorted(links.items())))


def _link_options(rules, limit=500):
    # Split rules normally continue later in reading order. Restrict joins
    # to sides using the same alphabets.
    def signature(rule):
        return tuple(tuple(sorted(set(t[0] for t in s))) for s in rule)

    sigs = [signature(r) for r in rules]
    pairs = [(i, j) for i in range(len(rules))
             for j in range(i + 1, len(rules)) if sigs[i] == sigs[j]]
    pairs.sort(key=lambda p: (p[1] - p[0], p))
    count = 0
    for i, j in pairs:
        yield {i: j}
        count += 1
        if count >= limit:
            return
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            i, j = pairs[a]
            k, l = pairs[b]
            if len({i, j, k, l}) != 4:
                continue
            yield {i: j, k: l}
            count += 1
            if count >= limit:
                return
    # Unusual layouts may continue in the opposite spatial order.
    for i, j in pairs:
        yield {j: i}
        count += 1
        if count >= limit:
            return


def _normal_solution(board, maps, bad, attempts, deadline):
    rules = [([_raw(t) for t in a], [_raw(t) for t in b])
             for a, b in board["rules"]]
    source = [_raw(t) for t in board["source"]]
    current = [_raw(t) for t in board["target"]]
    models = [("direct", {}), ("tree", {}), ("double", {})]
    models.extend(("double", links) for links in _link_options(rules))
    best = None

    for mode, links in models:
        if time.monotonic() > deadline:
            break
        key = _model_key(mode, links)
        if key in bad:
            continue
        output = _translate(rules, source, mode, links)
        if output is None or len(output) > len(current):
            continue
        if any(a[0] != b[0] for a, b in zip(output, current)):
            continue
        attempt = ("normal", tuple(output))
        if attempt in attempts:
            continue

        deltas = [0] * len(current)
        possible = True
        for i, (c, wanted) in enumerate(output):
            now = current[i][1]
            if wanted == now:
                continue
            table = maps.get(c, {})
            if now not in table or wanted not in table:
                possible = False
                break
            deltas[i] = (table[wanted] - table[now]) % 7
        if not possible:
            continue

        plan = _ring_plan(deltas, board["cursor"])
        intended = _translate(rules, source, mode, links, strict=True)
        penalty = 16 * (len(current) - len(output))
        penalty += {"direct": 0, "tree": 0.5, "double": 1}[mode]
        penalty += 2 * len(links)
        if intended != output:
            penalty += 12
        score = len(plan) + penalty
        if best is None or score < best[0]:
            best = (score, plan, key, attempt)

    return None if best is None else best[1:]


class _SearchEnd(Exception):
    pass


def _match_delta(pattern, sequence, pos, first=False):
    if first:
        pattern = pattern[:1]
    if pos + len(pattern) > len(sequence):
        return None
    delta = None
    for (c, v), (cc, vv) in zip(pattern, sequence[pos:]):
        if c != cc:
            return None
        d = (vv - v) % 7
        if delta is None:
            delta = d
        elif d != delta:
            return None
    return delta


def _bind(domains, variable, value):
    bit = 1 << value
    if not domains[variable] & bit:
        return None
    new = domains.copy()
    new[variable] = bit
    return new


def _terms(patterns, variables):
    return [(v, c, a) for v in variables for c, a in patterns[v]]


def _bind_terms(domains, terms, target, pos):
    if pos + len(terms) > len(target):
        return None
    new = domains.copy()
    for (v, c, a), (cc, b) in zip(terms, target[pos:]):
        if c != cc:
            return None
        bit = 1 << ((b - a) % 7)
        if not new[v] & bit:
            return None
        new[v] = bit
    return new


def _unify(domains, left, right):
    """Ground the small modular-equality components in a composition."""
    graph = {}
    for (u, c, a), (v, cc, b) in zip(left, right):
        if c != cc:
            return
        difference = (a - b) % 7
        if u == v:
            if difference:
                return
            continue
        graph.setdefault(u, []).append((v, difference))
        graph.setdefault(v, []).append((u, -difference))

    components = []
    seen = set()
    for root in graph:
        if root in seen:
            continue
        potential = {root: 0}
        stack = [root]
        seen.add(root)
        consistent = True
        while stack:
            u = stack.pop()
            for v, difference in graph[u]:
                pv = (potential[u] + difference) % 7
                if v in potential:
                    if potential[v] != pv:
                        consistent = False
                        break
                else:
                    potential[v] = pv
                    seen.add(v)
                    stack.append(v)
            if not consistent:
                break
        if not consistent:
            return
        choices = []
        for value in range(7):
            assignment = {v: (value + p) % 7
                          for v, p in potential.items()}
            if all(domains[v] & (1 << d)
                   for v, d in assignment.items()):
                choices.append(assignment)
        if not choices:
            return
        choices.sort(key=lambda a: sum(_distance(d) for d in a.values()))
        components.append(choices)

    if not components:
        yield domains
        return
    for combination in itertools.product(*components):
        new = domains.copy()
        for assignment in combination:
            for v, value in assignment.items():
                new[v] = 1 << value
        yield new


def _alter_solution(board, maps, bad, attempts, deadline):
    def numeric(group):
        return [(t["c"], maps[t["c"]][t["k"]]) for t in group]

    try:
        rules = [(numeric(a), numeric(b)) for a, b in board["rules"]]
        source = numeric(board["source"])
        target = numeric(board["target"])
    except KeyError:
        return None

    patterns = [s for rule in rules for s in rule]
    nr = len(rules)
    nv = 2 * nr
    left_deltas = [
        [_match_delta(patterns[2 * r], source, p)
         for p in range(len(source))]
        for r in range(nr)
    ]
    right_deltas = [
        [_match_delta(patterns[v], target, q)
         for q in range(len(target) + 1)]
        for v in range(nv)
    ]
    best = [None]

    def search(mode, links, stop_time, strict):
        key = _model_key(mode, links)
        if key in bad:
            return
        seconds = set(links.values())
        seen = set()
        nodes = [0]
        prior = {"direct": 0, "tree": 0.5, "double": 1}[mode]
        prior += 2 * len(links)

        def check():
            nodes[0] += 1
            if nodes[0] % 64 == 0:
                if nodes[0] > 90000 or time.monotonic() > stop_time:
                    raise _SearchEnd()

        def joined_vars(r, side):
            ans = [2 * r + side]
            if r in links:
                ans.append(2 * links[r] + side)
            return ans

        jl = [_terms(patterns, joined_vars(r, 0)) for r in range(nr)]
        jr = [_terms(patterns, joined_vars(r, 1)) for r in range(nr)]

        def ground(offsets):
            rr = []
            for r in range(nr):
                sides = []
                for side in (0, 1):
                    v = 2 * r + side
                    sides.append([(c, (a + offsets[v]) % 7)
                                  for c, a in patterns[v]])
                rr.append(tuple(sides))
            return rr

        def finish(domains):
            check()
            options = [
                sorted((d for d in range(7) if mask & (1 << d)),
                       key=lambda d: (_distance(d), d))
                for mask in domains
            ]
            if any(not a for a in options):
                return
            initial = (0,) * nv
            heap = [(sum(_distance(a[0]) for a in options), initial)]
            visited = {initial}
            trials = 0
            while heap and trials < 24:
                check()
                _, indices = heapq.heappop(heap)
                trials += 1
                offsets = [options[v][indices[v]] for v in range(nv)]
                predicted = _translate(ground(offsets), source, mode, links)
                if predicted == target:
                    state = tuple((patterns[v][0][1] + offsets[v]) % 7
                                  for v in range(nv))
                    attempt = ("alter", state)
                    if attempt not in attempts:
                        plan = _ring_plan(offsets, board["cursor"])
                        score = len(plan) + prior
                        if best[0] is None or score < best[0][0]:
                            best[0] = (score, plan, key, attempt)
                        return
                for v in range(nv):
                    if indices[v] + 1 >= len(options[v]):
                        continue
                    nxt = list(indices)
                    nxt[v] += 1
                    nxt = tuple(nxt)
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    cost = sum(_distance(options[j][nxt[j]])
                               for j in range(nv))
                    heapq.heappush(heap, (cost, nxt))

        def tree_outputs(r, q, domains):
            rv = 2 * r + 1
            for d in sorted(range(7), key=_distance):
                if not domains[rv] & (1 << d):
                    continue
                fixed = _bind(domains, rv, d)
                symbols = [(c, (a + d) % 7) for c, a in patterns[rv]]

                def expand(k, pos, ds):
                    check()
                    if k == len(symbols):
                        yield pos, ds
                        return
                    c, a = symbols[k]
                    prefix = ds.copy()
                    for s in range(nr):
                        lv, ov = 2 * s, 2 * s + 1
                        cc, b = patterns[lv][0]
                        if c != cc:
                            continue
                        need = (a - b) % 7
                        bit = 1 << need
                        if prefix[lv] & bit:
                            e = _bind(prefix, lv, need)
                            if pos <= len(target):
                                od = right_deltas[ov][pos]
                                if od is not None:
                                    e = _bind(e, ov, od)
                                    if e is not None:
                                        yield from expand(
                                            k + 1, pos + len(patterns[ov]), e)
                        prefix[lv] &= ~bit
                        if not prefix[lv]:
                            break

                yield from expand(0, q, fixed)

        def outputs(r, q, domains):
            check()
            if mode == "direct":
                v = 2 * r + 1
                d = right_deltas[v][q]
                if d is not None:
                    new = _bind(domains, v, d)
                    if new is not None:
                        yield q + len(patterns[v]), new
            elif mode == "tree":
                yield from tree_outputs(r, q, domains)
            else:
                intermediate = jr[r]
                for s in range(nr):
                    if s in seconds:
                        continue
                    if strict and len(intermediate) != len(jl[s]):
                        continue
                    ds = _bind_terms(domains, jr[s], target, q)
                    if ds is None:
                        continue
                    for new in _unify(ds, intermediate, jl[s]):
                        check()
                        yield q + len(jr[s]), new

        def walk(p, q, domains):
            check()
            if q > len(target):
                return
            if p >= len(source):
                if q == len(target):
                    finish(domains)
                return
            state = (p, q, tuple(domains))
            if state in seen:
                return
            seen.add(state)

            if best[0] is not None:
                lower = sum(min(_distance(d) for d in range(7)
                                if mask & (1 << d))
                            for mask in domains)
                if lower + prior > best[0][0]:
                    return

            prefix = domains.copy()
            for r in range(nr):
                if mode == "double" and r in seconds:
                    continue
                d = left_deltas[r][p]
                if d is None:
                    continue
                lv = 2 * r
                bit = 1 << d
                if prefix[lv] & bit:
                    ds = _bind(prefix, lv, d)
                    consumed = len(patterns[lv])
                    if mode == "double":
                        consumed = len(jl[r])
                        if strict:
                            ds = _bind_terms(ds, jl[r], source, p)
                    if ds is not None:
                        for qq, ee in outputs(r, q, ds):
                            walk(p + consumed, qq, ee)

                # In ordinary translations the first matching rule wins.
                # The relaxed pass also permits unavailable compositions.
                if strict or mode == "direct":
                    prefix[lv] &= ~bit
                    if not prefix[lv]:
                        break

        try:
            walk(0, 0, [127] * nv)
        except _SearchEnd:
            pass

    # Reject composition types whose alphabet flow cannot fit even the
    # first input/output symbol.
    def plausible(mode):
        sc, tc = source[0][0], target[0][0]
        for a, b in rules:
            if a[0][0] != sc:
                continue
            if mode == "direct":
                if b[0][0] == tc:
                    return True
            else:
                for aa, bb in rules:
                    if b[0][0] == aa[0][0] and bb[0][0] == tc:
                        return True
        return False

    modes = [m for m in ("direct", "tree", "double") if plausible(m)]
    for mode in modes:
        if time.monotonic() >= deadline:
            break
        search(mode, {}, min(deadline, time.monotonic() + 1.8), True)

    if best[0] is None:
        for mode in modes:
            if mode == "direct" or time.monotonic() >= deadline:
                continue
            search(mode, {}, min(deadline, time.monotonic() + 0.6), False)

    if best[0] is None and plausible("double"):
        for links in _link_options(rules):
            if time.monotonic() >= deadline:
                break
            search("double", links,
                   min(deadline, time.monotonic() + 0.22), True)
            if best[0] is not None:
                break

    return None if best[0] is None else best[0][1:]


def act(obs, available, memory):
    def emit(name):
        if name not in available:
            alternatives = [a for a in available if a != "RESET"]
            name = alternatives[0] if alternatives else available[0]
        return (name, None, None)

    # Predicted edits are deterministic, including their cursor movements.
    if memory.get("_plan"):
        return emit(memory["_plan"].pop(0))

    started = time.monotonic()
    board = _parse(np.asarray(obs))
    if board is None:
        return emit("ACTION4")

    maps = memory.setdefault("_maps", _initial_maps())

    geometry = tuple((t["x"], t["y"], t["c"]) for t in board["tiles"])
    invariant = (
        tuple(_raw(t) for t in board["source"]),
        tuple(_raw(t) for t in board["target"]) if board["alter"] else
        tuple(_raw(t) for rule in board["rules"] for side in rule for t in side)
    )
    signature = (geometry, board["alter"], invariant)
    if memory.get("_signature") != signature:
        memory["_signature"] = signature
        memory["_bad"] = []
        memory["_attempts"] = []
        memory.pop("_active", None)
        memory.pop("_calibration", None)

    # If the whole predicted winning plan ran but the episode continues,
    # that translation hypothesis was not the game's interpretation.
    active = memory.pop("_active", None)
    if active is not None:
        memory.setdefault("_bad", []).append(active)

    slots = board["slots"]
    cursor = board["cursor"]

    # Learn a previously unseen alphabet by observing one full edit cycle.
    cal = memory.get("_calibration")
    if cal is not None:
        sample = slots[cal["slot"]][cal["member"]]
        maps.setdefault(cal["color"], {})[sample["k"]] = (
            cal["base"] + cal["count"]) % 7
        if cal["count"] >= 7:
            memory.pop("_calibration", None)
        else:
            cal["count"] += 1
            return emit("ACTION2")

    unknown = []
    for i, group in enumerate(slots):
        for j, tile in enumerate(group):
            table = maps.get(tile["c"], {})
            if tile["k"] not in table or len(table) < 7:
                unknown.append((i, j, tile))
    if unknown:
        n = len(slots)
        i, j, tile = min(
            unknown,
            key=lambda z: min((z[0] - cursor) % n, (cursor - z[0]) % n)
        )
        if i != cursor:
            return emit(_move_to(cursor, i, n))
        table = maps.setdefault(tile["c"], {})
        base = table.get(tile["k"], 0)
        table[tile["k"]] = base
        memory["_calibration"] = {
            "slot": i, "member": j, "color": tile["c"],
            "base": base, "count": 1,
        }
        return emit("ACTION2")

    bad = set(memory.setdefault("_bad", []))
    attempts = set(memory.setdefault("_attempts", []))
    deadline = started + 8.3

    if board["alter"]:
        solution = _alter_solution(board, maps, bad, attempts, deadline)
    else:
        solution = _normal_solution(board, maps, bad, attempts, deadline)

    if solution is not None:
        plan, model, attempt = solution
        memory["_active"] = model
        memory["_attempts"].append(attempt)
        memory["_plan"] = plan[1:]
        return emit(plan[0])

    # A bounded exploratory edit provides another fully observed state
    # rather than assuming an unseen layout or restarting its progress.
    count = memory.get("_explore", 0)
    memory["_explore"] = count + 1
    if count % 4 == 3 and len(slots) > 1:
        return emit("ACTION4")
    return emit("ACTION2")

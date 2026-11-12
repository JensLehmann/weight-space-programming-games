import time
import numpy as np


def _item(obs, x, y):
    """Read a symbol at its 6x6 sprite origin."""
    if x < 0 or y < 0 or x + 6 > 64 or y + 6 > 64:
        return None

    a = obs[y + 1:y + 5, x + 1:x + 5]
    c = int(a[0, 0])
    if c == 4:
        return None

    if np.all(a == c):
        return (0, c)  # terminal

    if (np.all(a[0, :] == c) and np.all(a[-1, :] == c)
            and np.all(a[:, 0] == c) and np.all(a[:, -1] == c)
            and np.all(a[1:3, 1:3] == 4)):
        return (1, c)  # function call

    return None


def _selected(obs, x, y):
    return (np.all(obs[y, x:x + 6] == 0)
            and np.all(obs[y + 5, x:x + 6] == 0)
            and np.all(obs[y:y + 6, x] == 0)
            and np.all(obs[y:y + 6, x + 5] == 0))


def _read(obs):
    frames = []

    # Function boxes have height 10 and width 6*n + 4.
    for y in range(45):
        x = 0
        while x < 64:
            c = int(obs[y, x])
            end = x + 1
            while end < 64 and int(obs[y, end]) == c:
                end += 1

            width = end - x
            if c != 4 and width >= 10 and (width - 4) % 6 == 0:
                if (np.all(obs[y + 9, x:end] == c)
                        and np.all(obs[y:y + 10, x] == c)
                        and np.all(obs[y:y + 10, end - 1] == c)):
                    inside = obs[y + 1:y + 9, x + 1:end - 1]
                    # Reject rectangular pieces of the target's black backing.
                    if np.count_nonzero(inside == 4) >= inside.size * 0.35:
                        frames.append((y, x, (width - 4) // 6, c))
            x = end

    frames.sort()
    if not frames:
        return None

    upper = frames[0][0]
    targets = []
    for y in range(max(0, upper - 5)):
        for x in range(59):
            c = int(obs[y, x])
            if c in (4, 5):
                continue
            if not (np.all(obs[y, x:x + 6] == c)
                    and np.all(obs[y + 5, x:x + 6] == c)
                    and np.all(obs[y:y + 6, x] == c)
                    and np.all(obs[y:y + 6, x + 5] == c)):
                continue
            a = obs[y + 1:y + 5, x + 1:x + 5]
            if np.all((a == 5) | (a == c)):
                targets.append((y, x, c))

    targets.sort()
    if not targets:
        return None

    positions = []
    contents = []
    function_slots = []
    selection = None

    for y, x, n, c in frames:
        slots = []
        for j in range(n):
            sx, sy = x + 2 + 6 * j, y + 2
            slots.append(len(positions))
            positions.append((sx, sy))
            symbol = _item(obs, sx, sy)
            contents.append(symbol)
            if symbol is not None and _selected(obs, sx, sy):
                selection = (sx, sy)
        function_slots.append(tuple(slots))

    pool = []
    for y in range(54, 59):
        for x in range(59):
            symbol = _item(obs, x, y)
            if symbol is None:
                continue
            border = np.concatenate((
                obs[y, x:x + 6],
                obs[y + 5, x:x + 6],
                obs[y + 1:y + 5, x],
                obs[y + 1:y + 5, x + 5]
            ))
            if not np.all((border == 4) | (border == 0)):
                continue
            pool.append((symbol, x, y))
            if _selected(obs, x, y):
                selection = (x, y)

    return {
        "frames": frames,
        "target": tuple(t[2] for t in targets),
        "positions": positions,
        "contents": contents,
        "functions": function_slots,
        "pool": pool,
        "selection": selection,
    }


def _signature(layout):
    return (
        tuple(layout["frames"]),
        layout["target"],
        tuple(layout["contents"]),
        tuple(layout["pool"]),
    )


def _model(layout):
    keys = set(s for s in layout["contents"] if s is not None)
    keys.update(s for s, x, y in layout["pool"])
    keys.update((0, c) for c in layout["target"])
    types = sorted(keys)
    index = {s: i for i, s in enumerate(types)}

    color_function = {}
    for f, (_, _, _, c) in enumerate(layout["frames"]):
        color_function.setdefault(c, f)

    callees = [
        color_function.get(c, -1) if kind else -1
        for kind, c in types
    ]
    calls = [
        t for t, (kind, c) in enumerate(types)
        if kind == 1 and callees[t] >= 0
    ]

    counts = [0] * len(types)
    for symbol, x, y in layout["pool"]:
        counts[index[symbol]] += 1

    base = [
        -1 if symbol is None else index[symbol]
        for symbol in layout["contents"]
    ]

    return {
        "types": types,
        "callees": callees,
        "calls": calls,
        "terminal": {
            c: t for t, (kind, c) in enumerate(types) if kind == 0
        },
        "functions": layout["functions"],
        "target": layout["target"],
        "base": base,
        "counts": counts,
        "layout": layout,
    }


def _first_ok(model, f, color, assignment, counts, seen=0):
    """Optimistic first-output check, used to prune impossible calls."""
    bit = 1 << f
    if seen & bit:
        return False
    seen |= bit

    t = assignment[model["functions"][f][0]]
    if t >= 0:
        kind, c = model["types"][t]
        if kind == 0:
            return c == color
        g = model["callees"][t]
        return (g >= 0 and
                _first_ok(model, g, color, assignment, counts, seen))

    terminal = model["terminal"].get(color)
    if terminal is not None and counts[terminal] > 0:
        return True

    for call in model["calls"]:
        if counts[call] > 0:
            g = model["callees"][call]
            if _first_ok(model, g, color, assignment, counts, seen):
                return True
    return False


def _search(model, search=None, seconds=6.7):
    """
    Incremental depth-first synthesis.

    Work is a stack of slot IDs. Assignments persist across repeated calls.
    A bit mask detects cycles that cannot emit any output.
    """
    if search is None:
        search = {
            "assignment": list(model["base"]),
            "counts": list(model["counts"]),
            "branches": [],
            "state": (tuple(reversed(model["functions"][0])), 0, 1),
            "backtrack": False,
        }

    assignment = search["assignment"]
    counts = search["counts"]
    branches = search["branches"]
    target = model["target"]
    deadline = time.perf_counter() + seconds
    ticks = 0

    while True:
        ticks += 1
        if ticks % 128 == 0 and time.perf_counter() >= deadline:
            return "pending", search

        if search["backtrack"]:
            found = False
            while branches:
                branch = branches[-1]
                sid, state, candidates, cursor, chosen = branch

                if chosen >= 0:
                    assignment[sid] = -1
                    counts[chosen] += 1
                    branch[4] = -1

                while branch[3] < len(candidates):
                    t = candidates[branch[3]]
                    branch[3] += 1
                    if counts[t] <= 0:
                        continue

                    assignment[sid] = t
                    counts[t] -= 1

                    kind, c = model["types"][t]
                    if kind:
                        f = model["callees"][t]
                        if (f < 0 or not _first_ok(
                                model, f, target[state[1]],
                                assignment, counts)):
                            assignment[sid] = -1
                            counts[t] += 1
                            continue

                    branch[4] = t
                    search["state"] = state
                    search["backtrack"] = False
                    found = True
                    break

                if found:
                    break
                branches.pop()

            if not found:
                return "impossible", None
            continue

        work, output_index, mask = search["state"]
        work = list(work)
        failed = False
        unknown = False
        steps = 0

        while work:
            steps += 1
            if steps > 5000 or len(work) > 5000:
                failed = True
                break

            sid = work[-1]
            t = assignment[sid]
            if t < 0:
                unknown = True
                break

            work.pop()
            kind, color = model["types"][t]
            if kind == 0:
                if color != target[output_index]:
                    failed = True
                    break
                output_index += 1
                if output_index == len(target):
                    return "solved", list(assignment)
                mask = 0
            else:
                f = model["callees"][t]
                if f < 0 or mask & (1 << f):
                    failed = True
                    break
                mask |= 1 << f
                work.extend(reversed(model["functions"][f]))

        if failed or not unknown:
            search["backtrack"] = True
            continue

        state = (tuple(work), output_index, mask)
        candidates = []
        terminal = model["terminal"].get(target[output_index])
        if terminal is not None and counts[terminal] > 0:
            candidates.append(terminal)

        for t in model["calls"]:
            f = model["callees"][t]
            if counts[t] > 0 and not (mask & (1 << f)):
                candidates.append(t)

        branches.append([sid, state, candidates, 0, -1])
        search["backtrack"] = True


def _make_plan(model, assignment):
    layout = model["layout"]
    sources = {}
    for symbol, x, y in layout["pool"]:
        sources.setdefault(symbol, []).append((x, y))

    actions = []
    for sid, old in enumerate(model["base"]):
        t = assignment[sid]
        if old >= 0 or t < 0:
            continue

        symbol = model["types"][t]
        source = sources[symbol].pop()
        destination = layout["positions"][sid]
        actions.append(("ACTION6", source[1] + 2, source[0] + 2))
        actions.append((
            "ACTION6", destination[1] + 2, destination[0] + 2
        ))

    actions.append(("ACTION5", None, None))
    return actions


def _take_plan(memory):
    queue = memory["sb26_plan"]
    cursor = memory.get("sb26_cursor", 0)
    action = queue[cursor]
    cursor += 1
    if cursor == len(queue):
        memory.pop("sb26_plan", None)
        memory.pop("sb26_cursor", None)
    else:
        memory["sb26_cursor"] = cursor
    return action


def act(obs, available, memory):
    if "sb26_plan" in memory:
        return _take_plan(memory)

    if "sb26_model" in memory:
        model = memory["sb26_model"]
        search = memory.get("sb26_search")
    else:
        layout = _read(obs)
        if layout is None:
            # An undo is harmless at the initial state and also clears
            # transient arrangements when an episode starts partway through.
            if not memory.get("sb26_read_retry"):
                memory["sb26_read_retry"] = True
                return ("ACTION7", None, None)
            memory.pop("sb26_read_retry", None)
            return ("RESET", None, None)

        memory.pop("sb26_read_retry", None)

        # A perturbed start may already have a selected symbol.
        if layout["selection"] is not None:
            x, y = layout["selection"]
            return ("ACTION6", y + 2, x + 2)

        model = _model(layout)
        search = None

    status, result = _search(model, search)

    if status == "pending":
        memory["sb26_model"] = model
        memory["sb26_search"] = result
        # Target sprites are not clickable, so this safely preserves state.
        return ("ACTION6", 0, 0)

    memory.pop("sb26_model", None)
    memory.pop("sb26_search", None)

    if status == "solved":
        memory.pop("sb26_failed_layout", None)
        memory["sb26_plan"] = _make_plan(model, result)
        memory["sb26_cursor"] = 0
        return _take_plan(memory)

    signature = _signature(model["layout"])
    previous = memory.get("sb26_failed_layout")
    memory["sb26_failed_layout"] = signature

    if previous != signature:
        return ("ACTION7", None, None)

    # Undo history is exhausted. The preceding undo/no-op ensures that
    # RESET is a level reset rather than an initial full-game reset.
    memory.pop("sb26_failed_layout", None)
    return ("RESET", None, None)

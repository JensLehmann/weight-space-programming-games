import time
import numpy as np


def _border(a, x, y, w, h, color):
    return (
        np.all(a[y, x:x + w] == color)
        and np.all(a[y + h - 1, x:x + w] == color)
        and np.all(a[y:y + h, x] == color)
        and np.all(a[y:y + h, x + w - 1] == color)
    )


def _token(a, x, y):
    """Token code: terminal=color, function call=16+color, empty=-1."""
    if x < 0 or y < 0 or x + 5 >= 64 or y + 5 >= 64:
        return -1
    c = int(a[y + 1, x + 1])
    if c == 4 or not _border(a, x + 1, y + 1, 4, 4, c):
        return -1
    middle = a[y + 2:y + 4, x + 2:x + 4]
    if np.all(middle == c):
        return c
    if np.all(middle == 4):
        return 16 + c
    return -1


def _read_board(a):
    frames = []
    for y in range(45):
        for x in range(55):
            c = int(a[y, x])
            if c == 4 or int(a[y + 1, x]) != c:
                continue
            max_n = min(9, (64 - x - 4) // 6)
            for n in range(1, max_n + 1):
                w = 6 * n + 4
                if int(a[y, x + w - 1]) != c:
                    continue
                if not _border(a, x, y, w, 10, c):
                    continue
                if not np.all(a[y + 1, x + 1:x + w - 1] == 4):
                    continue
                if not np.all(a[y + 8, x + 1:x + w - 1] == 4):
                    continue
                frames.append((x, y, n, c))

    frames.sort(key=lambda f: (f[1], f[0]))
    if not frames:
        return None

    targets = []
    for y in range(max(0, frames[0][1] - 5)):
        for x in range(59):
            c = int(a[y, x])
            if c == 5:
                continue
            if not np.all(a[y + 1:y + 5, x + 1:x + 5] == 5):
                continue
            if _border(a, x, y, 6, 6, c):
                targets.append((y, x, c))
    targets.sort()
    target = [t[2] for t in targets]
    if not target:
        return None

    positions = []
    programs = []
    current = []
    for x, y, n, c in frames:
        indices = []
        for j in range(n):
            px, py = x + 2 + 6 * j, y + 2
            indices.append(len(positions))
            positions.append((px, py))
            current.append(_token(a, px, py))
        programs.append(indices)

    inventory = []
    for y in range(54, 59):
        for x in range(59):
            code = _token(a, x, y)
            if code >= 0:
                inventory.append((x, y, code))

    selected = None
    for i, (x, y) in enumerate(positions):
        if current[i] >= 0 and _border(a, x, y, 6, 6, 0):
            selected = (x, y)
            break
    if selected is None:
        for x, y, code in inventory:
            if _border(a, x, y, 6, 6, 0):
                selected = (x, y)
                break

    return {
        "frames": frames,
        "programs": programs,
        "target": target,
        "positions": positions,
        "current": current,
        "inventory": inventory,
        "selected": selected,
        "signature": repr((frames, target)),
    }


class _SearchTimeout(Exception):
    pass


def _synthesize(board, locks, fixed_all, change_limit, banned, deadline):
    programs = board["programs"]
    target = board["target"]
    current = board["current"]
    nslots = len(current)

    destinations = {}
    for f, frame in enumerate(board["frames"]):
        destinations.setdefault(frame[3], f)

    counts = [0] * 32
    for x, y, code in board["inventory"]:
        counts[code] += 1

    if fixed_all:
        values = current[:]
    else:
        values = [-1] * nslots
        for i, code in enumerate(current):
            if i in locks:
                values[i] = code
            elif code >= 0:
                counts[code] += 1

    call_types = [
        code for code in range(16, 32)
        if counts[code] and code - 16 in destinations
    ]
    banned_set = {tuple(v) for v in banned}
    nodes = [0]

    def dfs(stack, p, elapsed, changes):
        nodes[0] += 1
        if nodes[0] % 512 == 0 and time.monotonic() >= deadline:
            raise _SearchTimeout()

        while stack:
            if elapsed > 960:
                return None

            f, pc, entry_p = stack[-1]
            if pc >= len(programs[f]):
                if len(stack) == 1:
                    return None
                # Return to the calling instruction, then advance past it.
                parent = stack[-2]
                stack = stack[:-2] + (
                    (parent[0], parent[1] + 1, parent[2]),
                )
                elapsed += 16
                continue

            slot = programs[f][pc]
            code = values[slot]

            if code < 0:
                candidates = []
                terminal = target[p]
                if counts[terminal] > 0:
                    candidates.append(terminal)
                candidates.extend(c for c in call_types if counts[c] > 0)

                old = current[slot]
                if old in candidates:
                    candidates.remove(old)
                    candidates.insert(0, old)

                for candidate in candidates:
                    extra = int(old >= 0 and old != candidate)
                    if changes + extra > change_limit:
                        continue
                    values[slot] = candidate
                    counts[candidate] -= 1
                    result = dfs(stack, p, elapsed, changes + extra)
                    counts[candidate] += 1
                    values[slot] = -1
                    if result is not None:
                        return result
                return None

            if code < 16:
                if code != target[p]:
                    return None
                elapsed += (16 if pc == 0 else 11) + 13
                p += 1
                if p == len(target):
                    if elapsed + 16 > 980:
                        return None
                    result = values[:]
                    if tuple(result) not in banned_set:
                        return result
                    return None
                stack = stack[:-1] + ((f, pc + 1, entry_p),)
                continue

            dest = destinations.get(code - 16)
            if dest is None:
                return None

            # Reproduce the game's explicit recursion guard.
            if pc == 0 and len(stack) > 1 and stack[-2][1] == 0:
                if any(af == f and apc == 0 for af, apc, ep in stack[:-1]):
                    return None

            # Also reject recursion that cannot emit anything.
            if any(af == dest and ep == p for af, apc, ep in stack):
                return None

            elapsed += 16
            stack = stack + ((dest, 0, p),)

        return None

    return dfs(((0, 0, 0),), 0, 0, 0)


def _click(position):
    x, y = position
    return ("ACTION6", int(y + 2), int(x + 2))


def _reset(memory):
    resets = int(memory.get("resets", 0)) + 1
    memory.clear()
    memory["resets"] = resets
    return ("RESET", None, None)


def act(obs, available, memory):
    started = time.monotonic()
    a = np.asarray(obs)
    board = _read_board(a)
    if board is None:
        return _reset(memory)

    signature = board["signature"]
    if memory.get("signature") != signature:
        resets = memory.get("resets", 0)
        memory.clear()
        memory["resets"] = resets
        memory["signature"] = signature
        memory["locks"] = []
        memory["banned"] = []

    positions = board["positions"]
    current = board["current"]
    selected = board["selected"]
    locks = set(memory.get("locks", []))

    # An ACTION5 that returned to this board did not finish the level.
    tested = memory.pop("tested", None)
    if tested is not None:
        memory.setdefault("banned", []).append(tested)
        memory.pop("goal", None)

    transfer = memory.get("transfer")
    if transfer is not None:
        if transfer["stage"] == "selected":
            first = tuple(transfer["first"])
            if selected == first:
                transfer["stage"] = "sent"
                return _click(tuple(transfer["second"]))

            # The attempted selection reveals an immovable program piece.
            if transfer["first_slot"] >= 0:
                locks.add(transfer["first_slot"])
            memory["locks"] = sorted(locks)
            memory.pop("transfer", None)
            memory.pop("goal", None)

        else:
            target_slot = transfer["target"]
            if current[target_slot] == transfer["wanted"]:
                memory.pop("transfer", None)
            else:
                if (
                    transfer["second_slot"] >= 0
                    and transfer["second_code"] >= 0
                ):
                    locks.add(transfer["second_slot"])
                memory["locks"] = sorted(locks)
                memory.pop("transfer", None)
                memory.pop("goal", None)

    # Clear selections inherited from a perturbed starting state or a failed move.
    if selected is not None:
        return _click(selected)

    goal = memory.get("goal")
    if goal is None:
        deadline = started + 8.0
        banned = memory.get("banned", [])
        try:
            # Usually all existing program pieces can simply be preserved.
            goal = _synthesize(
                board, locks, True, 0, banned, deadline
            )
            if goal is None:
                # Repair displaced pieces without resetting the whole puzzle.
                goal = _synthesize(
                    board, locks, False, 2, banned, deadline
                )
            if goal is None:
                goal = _synthesize(
                    board, locks, False, len(current), banned, deadline
                )
        except _SearchTimeout:
            goal = None

        if goal is None:
            return _reset(memory)
        memory["goal"] = goal

    wrong = [
        i for i, wanted in enumerate(goal)
        if wanted >= 0 and current[i] != wanted
    ]
    if not wrong:
        memory["tested"] = list(goal)
        memory.pop("goal", None)
        return ("ACTION5", None, None)

    # Find a token that is not already satisfying a required slot.
    choices = []
    for target_slot in wrong:
        wanted = goal[target_slot]
        tx, ty = positions[target_slot]

        for x, y, code in board["inventory"]:
            if code == wanted:
                priority = 0 if current[target_slot] < 0 else 2
                choices.append((
                    priority, target_slot, (x, y), -1, code
                ))

        for source_slot, code in enumerate(current):
            if source_slot == target_slot or code != wanted:
                continue
            if source_slot in locks:
                continue
            if goal[source_slot] == code:
                continue

            priority = 1 if current[target_slot] < 0 else 3
            if (
                current[target_slot] >= 0
                and goal[source_slot] == current[target_slot]
            ):
                priority = -1  # One swap fixes both positions.
            choices.append((
                priority, target_slot, positions[source_slot],
                source_slot, code
            ))

    if not choices:
        return _reset(memory)

    choices.sort(key=lambda c: (c[0], c[1], c[3]))
    _, target_slot, source_pos, source_slot, source_code = choices[0]
    target_pos = positions[target_slot]
    target_code = current[target_slot]

    if target_code >= 0:
        # Select the program token first. This tests whether it is fixed,
        # and swapping with the donor then puts the desired token here.
        first = target_pos
        second = source_pos
        first_slot = target_slot
        second_slot = source_slot
        second_code = source_code
    else:
        first = source_pos
        second = target_pos
        first_slot = source_slot
        second_slot = target_slot
        second_code = -1

    memory["transfer"] = {
        "stage": "selected",
        "first": list(first),
        "second": list(second),
        "first_slot": first_slot,
        "second_slot": second_slot,
        "second_code": second_code,
        "target": target_slot,
        "wanted": goal[target_slot],
    }
    return _click(first)

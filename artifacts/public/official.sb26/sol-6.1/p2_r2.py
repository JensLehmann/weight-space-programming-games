import time
from collections import Counter
import numpy as np


def _border_is(a, x, y, w, h, color):
    return (
        np.all(a[y, x:x + w] == color)
        and np.all(a[y + h - 1, x:x + w] == color)
        and np.all(a[y:y + h, x] == color)
        and np.all(a[y:y + h, x + w - 1] == color)
    )


def _piece(a, x, y):
    """Read a six-pixel piece. IDs 0..15 are literals, 16..31 are calls."""
    if x < 0 or y < 0 or x + 6 > 64 or y + 6 > 64:
        return None
    c = int(a[y + 1, x + 1])
    if c == 4:
        return None
    if not _border_is(a, x + 1, y + 1, 4, 4, c):
        return None
    middle = a[y + 2:y + 4, x + 2:x + 4]
    return c if np.all(middle == c) else 16 + c


def _read_board(a):
    targets = []
    # Targets have six-pixel borders and the dark target-panel background.
    for y in range(0, 20):
        for x in range(0, 59):
            c = int(a[y, x])
            if c in (4, 5):
                continue
            if not _border_is(a, x, y, 6, 6, c):
                continue
            inside = a[y + 1:y + 5, x + 1:x + 5]
            if np.all(inside == 5):
                targets.append((y, x, c))
    targets.sort()
    word = tuple(t[2] for t in targets)

    frames = []
    # Program boxes are ten pixels tall and 6*n+4 pixels wide.
    for y in range(8, 44):
        x = 0
        while x < 64:
            c = int(a[y, x])
            end = x + 1
            while end < 64 and int(a[y, end]) == c:
                end += 1
            width = end - x
            if (
                c not in (4, 5)
                and width >= 10
                and (width - 4) % 6 == 0
                and _border_is(a, x, y, width, 10, c)
            ):
                frames.append((x, y, (width - 4) // 6, c))
            x = end
    frames.sort(key=lambda f: (f[1], f[0]))

    slots = []
    groups = []
    fixed = []
    for x, y, n, c in frames:
        group = []
        for i in range(n):
            sx, sy = x + 2 + 6 * i, y + 2
            group.append(len(slots))
            slots.append((sx, sy))
            value = _piece(a, sx, sy)
            fixed.append(-1 if value is None else value)
        groups.append(tuple(group))

    pool = []
    for y in range(54, 59):
        for x in range(59):
            value = _piece(a, x, y)
            if value is not None:
                pool.append((x, y, value))

    selected = None
    positions = [(x, y) for x, y, value in pool]
    positions.extend(
        slots[i] for i, value in enumerate(fixed) if value != -1
    )
    for x, y in positions:
        if _border_is(a, x, y, 6, 6, 0):
            selected = (x, y)
            break

    return word, frames, groups, slots, fixed, pool, selected


def _synthesize(word, frames, groups, fixed, pool, variant=0):
    """
    Assign pieces while interpreting the program. A successful expansion only
    needs to reach the final target: subsequent instructions are not executed.
    """
    if not word or not frames:
        return None

    destination = {}
    for fi, frame in enumerate(frames):
        destination.setdefault(frame[3], fi)

    counts = [0] * 32
    for x, y, value in pool:
        counts[value] += 1

    assignment = list(fixed)
    call_types = [
        value for value in range(16, 32)
        if counts[value] and value - 16 in destination
    ]
    if variant % 2:
        call_types.reverse()

    # Every target color must exist as a literal somewhere.
    literal_colors = {
        value for value in fixed if 0 <= value < 16
    }
    literal_colors.update(i for i in range(16) if counts[i])
    if any(c not in literal_colors for c in word):
        return None

    deadline = time.perf_counter() + 7.2
    failed = set()
    nodes = 0
    aborted = False
    max_cost = 920

    def search(fi, si, stack, output, trail, cost):
        nonlocal nodes, aborted
        nodes += 1
        if nodes > 300000:
            aborted = True
            return None
        if nodes % 128 == 0 and time.perf_counter() >= deadline:
            aborted = True
            return None

        # Execute already-defined instructions until a new choice is needed.
        while True:
            if cost > max_cost:
                return None
            if output == len(word):
                return tuple(assignment)

            if si >= len(groups[fi]):
                if not stack:
                    return None
                fi, si = stack[-1]
                stack = stack[:-1]
                cost += 16
                continue

            index = groups[fi][si]
            bit = 1 << index
            if trail & bit:
                # Revisiting an instruction without producing a literal is
                # an infinite, nonproductive call cycle.
                return None

            value = assignment[index]
            if value == -1:
                break

            trail |= bit
            if value < 16:
                if value != word[output]:
                    return None
                output += 1
                si += 1
                trail = 0
                cost += 29
            else:
                child = destination.get(value - 16)
                if child is None:
                    return None
                stack = stack + ((fi, si + 1),)
                fi, si = child, 0
                cost += 16

        state = (
            fi, si, stack, output, trail, cost, tuple(assignment)
        )
        if state in failed:
            return None

        candidates = []
        literal = word[output]
        if counts[literal]:
            candidates.append(literal)

        # Prefer callees whose first instruction already matches the target.
        calls = [v for v in call_types if counts[v]]

        def call_priority(value):
            child = destination[value - 16]
            first = assignment[groups[child][0]]
            if first == literal:
                rank = 0
            elif first == -1:
                rank = 1
            elif first >= 16:
                rank = 2
            else:
                rank = 3
            return rank

        calls.sort(key=call_priority)
        candidates.extend(calls)

        for value in candidates:
            assignment[index] = value
            counts[value] -= 1
            result = search(fi, si, stack, output, trail, cost)
            counts[value] += 1
            assignment[index] = -1
            if result is not None:
                return result
            if aborted:
                return None

        if len(failed) < 65000:
            failed.add(state)
        return None

    return search(0, 0, (), 0, 0, 0)


def _click(position):
    x, y = position
    return "ACTION6", int(y + 2), int(x + 2)


def _restart(memory):
    memory.pop("plan", None)
    memory.pop("awaiting_test", None)
    memory["retries"] = memory.get("retries", 0) + 1
    return "RESET", None, None


def act(obs, available, memory):
    a = np.asarray(obs)
    word, frames, groups, slots, fixed, pool, selected = _read_board(a)

    signature = (word, tuple(frames))
    if memory.get("signature") != signature:
        memory.clear()
        memory["signature"] = signature

    if not word or not frames:
        # Normally only needed for an unusual interrupted starting state.
        if memory.get("parse_recovery", 0) == 0:
            memory["parse_recovery"] = 1
            return "ACTION7", None, None
        return _restart(memory)

    # If an executed solution returns control in this same level, recover
    # rather than wasting the remainder of the episode repeating the test.
    if memory.pop("awaiting_test", False):
        return _restart(memory)

    if "plan" not in memory:
        solution = _synthesize(
            word, frames, groups, fixed, pool,
            memory.get("retries", 0)
        )
        if solution is None:
            # Undoing can cheaply repair an episode begun after a bad move.
            undos = memory.get("recovery_undos", 0)
            if undos < 2:
                memory["recovery_undos"] = undos + 1
                return "ACTION7", None, None
            memory["recovery_undos"] = 0
            return _restart(memory)

        memory["plan"] = [
            (slots[i][0], slots[i][1], value)
            for i, value in enumerate(solution)
            if value != -1
        ]

    for x, y, desired in memory["plan"]:
        present = _piece(a, x, y)
        if present == desired:
            continue

        if present is not None:
            # This should not occur during normal execution; the synthesis
            # treated all initially occupied program cells as fixed.
            return _restart(memory)

        sources = [
            (px, py) for px, py, value in pool if value == desired
        ]
        if not sources:
            return _restart(memory)

        if selected in sources:
            return _click((x, y))

        source = sources[0]
        if selected is not None and selected != source:
            if selected[1] <= 53:
                # Selecting another piece from an in-box selection would
                # swap them. Deselect first instead.
                return _click(selected)
            # Pool-to-pool selection changes the selected piece directly.
        return _click(source)

    # All visited instructions are ready. Unvisited empty cells are harmless.
    memory["awaiting_test"] = True
    return "ACTION5", None, None

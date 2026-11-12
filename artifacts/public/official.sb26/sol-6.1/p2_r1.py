import time
from collections import Counter

import numpy as np


class _SearchTimeout(Exception):
    pass


def _tile_at(a, x, y):
    """Read a six-pixel tile sprite from its four-pixel visible interior."""
    if x < 0 or y < 0 or x + 5 >= 64 or y + 5 >= 64:
        return None
    p = a[y + 1:y + 5, x + 1:x + 5]
    c = int(p[0, 0])
    if c == 4:
        return None
    if not (
        np.all(p[0, :] == c)
        and np.all(p[-1, :] == c)
        and np.all(p[:, 0] == c)
        and np.all(p[:, -1] == c)
    ):
        return None
    middle = p[1:3, 1:3]
    if np.all(middle == c):
        return (0, c)  # literal
    if np.all(middle == 4):
        return (1, c)  # function call
    return None


def _selected_at(a, x, y):
    if x < 0 or y < 0 or x + 5 >= 64 or y + 5 >= 64:
        return False
    return bool(
        np.all(a[y, x:x + 6] == 0)
        and np.all(a[y + 5, x:x + 6] == 0)
        and np.all(a[y:y + 6, x] == 0)
        and np.all(a[y:y + 6, x + 5] == 0)
    )


def _read_scene(a):
    targets = []
    # Targets have a six-by-six colored border and black interiors.
    for y in range(0, 48):
        for x in range(0, 59):
            c = int(a[y, x])
            if c in (4, 5):
                continue
            if (
                np.all(a[y, x:x + 6] == c)
                and np.all(a[y + 5, x:x + 6] == c)
                and np.all(a[y:y + 6, x] == c)
                and np.all(a[y:y + 6, x + 5] == c)
                and np.all(a[y + 1:y + 5, x + 1:x + 5] == 5)
            ):
                targets.append((y, x, c))
    targets.sort()
    target = tuple(t[2] for t in targets)

    frames = []
    # Function frames are ten pixels high, with width 6*capacity + 4.
    for y in range(8, 44):
        x = 0
        while x < 64:
            c = int(a[y, x])
            if c == 4:
                x += 1
                continue
            end = x + 1
            while end < 64 and int(a[y, end]) == c:
                end += 1
            if end - x >= 10:
                for left in range(x, end - 9):
                    for n in range(1, 8):
                        width = 6 * n + 4
                        right = left + width - 1
                        if right >= end:
                            break
                        if (
                            np.all(a[y + 9, left:right + 1] == c)
                            and np.all(a[y:y + 10, left] == c)
                            and np.all(a[y:y + 10, right] == c)
                        ):
                            frames.append((left, y, n, c))
            x = end
    frames = sorted(set(frames), key=lambda f: (f[1], f[0]))

    slots = []
    items = {}
    for x, y, n, c in frames:
        for i in range(n):
            pos = (x + 2 + 6 * i, y + 2)
            slots.append(pos)
            tile = _tile_at(a, *pos)
            if tile is not None:
                items[pos] = tile

    # All loose tiles lie below the energy bar.
    for y in range(54, 59):
        for x in range(0, 59):
            tile = _tile_at(a, x, y)
            if tile is not None:
                items[(x, y)] = tile

    selected = None
    for pos in items:
        if _selected_at(a, *pos):
            selected = pos
            break

    return target, frames, slots, items, selected


def _search_program(
    target, frames, slots, items, fixed_positions,
    change_budget, deadline
):
    """Symbolically execute the program, assigning unknown slots on demand."""
    if not target or not frames:
        return None

    types = set(items.values())
    types.update((0, c) for c in target)
    types.update((1, f[3]) for f in frames)
    types = sorted(types)
    type_id = {t: i for i, t in enumerate(types)}

    entries = []
    nxt = []
    offset = 0
    color_entry = {}
    for x, y, n, color in frames:
        entries.append(offset)
        color_entry.setdefault(color, offset)
        for j in range(n):
            nxt.append(offset + j + 1 if j + 1 < n else -1)
        offset += n

    current = [
        type_id[items[p]] if p in items else -1
        for p in slots
    ]
    values = [-1] * len(slots)
    fixed = set(fixed_positions)
    for i, pos in enumerate(slots):
        if pos in fixed and pos in items:
            values[i] = current[i]

    remaining = [0] * len(types)
    for pos, tile in items.items():
        if pos not in fixed:
            remaining[type_id[tile]] += 1

    call_entry = [-1] * len(types)
    references = []
    for i, (kind, color) in enumerate(types):
        if kind == 1 and color in color_entry:
            call_entry[i] = color_entry[color]
            references.append(i)
    references.sort(
        key=lambda t: (call_entry[t] == entries[0], call_entry[t])
    )
    literal_ids = [type_id[(0, c)] for c in target]
    ticks = 0

    def walk(node, stack, out_index, seen, changes, fuel):
        nonlocal ticks

        while True:
            ticks += 1
            if ticks % 256 == 0 and time.perf_counter() >= deadline:
                raise _SearchTimeout
            if fuel > 915:
                return None

            while node < 0:
                if not stack:
                    return None
                node = stack[-1]
                stack = stack[:-1]
                fuel += 12

            if node in seen:
                # A repeated program location without intervening output
                # cannot make progress.
                return None

            value = values[node]
            if value < 0:
                candidates = []
                lit = literal_ids[out_index]
                if remaining[lit] > 0:
                    candidates.append(lit)
                for ref in references:
                    if remaining[ref] > 0:
                        candidates.append(ref)

                old = current[node]
                if old in candidates:
                    candidates.remove(old)
                    candidates.insert(0, old)

                for candidate in candidates:
                    extra = int(old >= 0 and candidate != old)
                    if changes + extra > change_budget:
                        continue
                    values[node] = candidate
                    remaining[candidate] -= 1
                    result = walk(
                        node, stack, out_index, seen.copy(),
                        changes + extra, fuel
                    )
                    remaining[candidate] += 1
                    values[node] = -1
                    if result is not None:
                        return result
                return None

            seen.add(node)
            kind, color = types[value]
            if kind == 0:
                if color != target[out_index]:
                    return None
                out_index += 1
                fuel += 26
                if out_index == len(target):
                    return [
                        None if v < 0 else types[v]
                        for v in values
                    ]
                seen = set()
                node = nxt[node]
            else:
                entry = call_entry[value]
                if entry < 0:
                    return None
                stack = stack + (nxt[node],)
                node = entry
                fuel += 16

    try:
        if time.perf_counter() >= deadline:
            return None
        return walk(entries[0], (), 0, set(), 0, 0)
    except _SearchTimeout:
        return None


def _make_plan(target, frames, slots, items, locked, movable, deadline):
    occupied = {p for p in slots if p in items}

    # First keep every apparently fixed existing tile. This also gives
    # a particularly small repair plan for most perturbed observations.
    conservative = locked | (occupied - movable)
    first_deadline = min(deadline, time.perf_counter() + 3.0)
    plan = _search_program(
        target, frames, slots, items, conservative,
        len(slots), first_deadline
    )
    if plan is not None:
        return plan

    # If some existing placements are wrong, allow them to change.
    # Prefer solutions disturbing only a few occupied slots.
    for budget, allowance in ((0, 0.55), (1, 0.9), (2, 1.2)):
        if time.perf_counter() >= deadline:
            return None
        plan = _search_program(
            target, frames, slots, items, locked, budget,
            min(deadline, time.perf_counter() + allowance)
        )
        if plan is not None:
            return plan

    if time.perf_counter() < deadline:
        return _search_program(
            target, frames, slots, items, locked,
            len(slots), deadline
        )
    return None


def _fallback(available, state):
    state["plan"] = None
    state["pending"] = None
    failures = state.get("fallbacks", 0) + 1
    state["fallbacks"] = failures

    # Undo is useful for unfamiliar perturbed starts, and costs no energy.
    if failures <= 4 and "ACTION7" in available:
        state["locked"] = set()
        state["movable"] = set()
        return "ACTION7", None, None
    if "RESET" in available:
        state.clear()
        return "RESET", None, None
    if "ACTION7" in available:
        return "ACTION7", None, None
    return "ACTION5", None, None


def act(obs, available, memory):
    started = time.perf_counter()
    a = np.asarray(obs)
    state = memory.setdefault("_sb26_player", {})

    target, frames, slots, items, selected = _read_scene(a)
    if not target or not frames:
        return _fallback(available, state)

    signature = (target, tuple(frames))
    if state.get("signature") != signature:
        state.clear()
        state["signature"] = signature
        state["locked"] = set()
        state["movable"] = set()

    locked = set(state.get("locked", set()))
    movable = set(state.get("movable", set()))
    slot_set = set(slots)

    if selected is not None:
        movable.add(selected)
        locked.discard(selected)

    pending = state.pop("pending", None)
    if pending:
        kind = pending[0]
        if kind == "select":
            source = pending[1]
            if selected == source:
                movable.add(source)
            elif source in slot_set and source in items:
                # A fixed initial tile does not respond to selection.
                locked.add(source)
                movable.discard(source)
                state["plan"] = None

        elif kind == "place":
            source, dest, wanted = pending[1:]
            if items.get(dest) == wanted and selected != source:
                movable.add(dest)
                if source in items:
                    movable.add(source)
            elif selected == source:
                if dest in slot_set and dest in items:
                    locked.add(dest)
                    movable.discard(dest)
                state["plan"] = None

        elif kind == "run":
            # Normally the episode has ended before this branch is reached.
            # If execution failed, discard the program and inspect again.
            state["plan"] = None

    state["locked"] = locked
    state["movable"] = movable

    plan = state.get("plan")
    if plan is not None:
        if len(plan) != len(slots):
            plan = None
        else:
            for i, pos in enumerate(slots):
                if (
                    pos in locked and plan[i] is not None
                    and items.get(pos) != plan[i]
                ):
                    plan = None
                    break

    if plan is None:
        plan = _make_plan(
            target, frames, slots, items, locked, movable,
            started + 8.1
        )
        state["plan"] = plan

    if plan is None:
        return _fallback(available, state)

    required = {
        pos: wanted
        for pos, wanted in zip(slots, plan)
        if wanted is not None
    }
    mismatches = [
        (pos, wanted)
        for pos, wanted in required.items()
        if items.get(pos) != wanted
    ]

    if not mismatches:
        state["fallbacks"] = 0
        state["pending"] = ("run",)
        return "ACTION5", None, None

    # Find a source not already satisfying a required slot. Swapping two
    # misplaced tiles fixes one destination without needing a spare slot.
    best = None
    for dest, wanted in mismatches:
        for source, tile in items.items():
            if tile != wanted or source == dest or source in locked:
                continue
            if required.get(source) == tile:
                continue

            source_cost = (
                -20 if source == selected else
                0 if source[1] > 53 else
                1 if source in movable else 4
            )
            dest_cost = (
                0 if dest not in items else
                1 if dest in movable else 3
            )
            score = source_cost + dest_cost
            candidate = (score, dest[1], dest[0], source, dest, wanted)
            if best is None or candidate[:3] < best[:3]:
                best = candidate

    if best is None:
        return _fallback(available, state)

    _, _, _, source, dest, wanted = best
    state["fallbacks"] = 0

    if selected is not None and selected != source:
        # Clicking another in-frame tile while holding a tile would swap
        # them, so explicitly clear an unwanted selection first.
        state["pending"] = ("clear",)
        return "ACTION6", int(selected[1] + 2), int(selected[0] + 2)

    if selected == source:
        state["pending"] = ("place", source, dest, wanted)
        return "ACTION6", int(dest[1] + 2), int(dest[0] + 2)

    state["pending"] = ("select", source)
    return "ACTION6", int(source[1] + 2), int(source[0] + 2)

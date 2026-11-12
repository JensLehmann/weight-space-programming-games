import heapq
import time
from collections import deque

import numpy as np


# Spell patterns, in row-major order on the 3x3 panel.
SPELLS = {
    "fire": (1 << 1) | (1 << 4) | (1 << 7),
    "teleport": (1 << 0) | (1 << 1) | (1 << 4),
    "resize": (1 << 1) | (1 << 3) | (1 << 5) | (1 << 7),
}

# Directions use the game's ordering: up, down, left, right.
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
DIR_ACTION = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")


def _inside_button(x, y, buttons):
    return any(bx <= x < bx + 10 and by <= y < by + 10
               for bx, by in buttons)


def _find_buttons(a):
    buttons = []
    abilities = set()
    for y in range(55):
        for x in range(55):
            if (a[y, x] != 3 or a[y, x + 9] != 3 or
                    a[y + 9, x] != 3 or a[y + 9, x + 9] != 3):
                continue
            if not (np.all(a[y, x:x + 10] == 3) and
                    np.all(a[y + 9, x:x + 10] == 3) and
                    np.all(a[y:y + 10, x] == 3) and
                    np.all(a[y:y + 10, x + 9] == 3)):
                continue
            interior = a[y + 1:y + 9, x + 1:x + 9]
            counts = {
                "fire": int(np.count_nonzero(interior == 6)),
                "teleport": int(np.count_nonzero(interior == 11)),
                "resize": int(np.count_nonzero(interior == 15)),
            }
            name = max(counts, key=counts.get)
            if counts[name] >= 4:
                buttons.append((x, y))
                abilities.add(name)
    return buttons, abilities


def _find_goal_and_player(a):
    base_goal = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ], dtype=np.int8)

    goals = []
    for k in range(4):
        t = np.rot90(base_goal, -k)
        h, w = t.shape
        for y in range(65 - h):
            for x in range(65 - w):
                if a[y, x] == t[0, 0] and np.array_equal(a[y:y+h, x:x+w], t):
                    goals.append((x, y, w, h))

    if not goals:
        return None, None

    goal = goals[0]
    gx, gy, gw, gh = goal
    base_player = np.array([[10, 10], [9, 9]], dtype=np.int8)
    # Clockwise rotations correspond to down, left, up, right.
    directions = (1, 2, 0, 3)
    candidates = []
    for scale in (2, 1):
        for k in range(4):
            t = np.rot90(base_player, -k)
            if scale == 2:
                t = np.repeat(np.repeat(t, 2, axis=0), 2, axis=1)
            n = 2 * scale
            for y in range(65 - n):
                for x in range(65 - n):
                    if x < gx + gw and x + n > gx and y < gy + gh and y + n > gy:
                        continue
                    if a[y, x] == t[0, 0] and np.array_equal(a[y:y+n, x:x+n], t):
                        candidates.append((x, y, scale, directions[k]))

    return goal, candidates[0] if candidates else None


def _find_switches(a, buttons):
    result = []
    for y in range(61):
        for x in range(61):
            if _inside_button(x, y, buttons):
                continue
            if a[y, x] != 6:
                continue
            patch = a[y:y+4, x:x+4]
            center = int(patch[1, 1])
            if center not in (12, 13):
                continue
            if (np.all(patch[0, :] == 6) and
                    np.all(patch[3, :] == 6) and
                    np.all(patch[:, 0] == 6) and
                    np.all(patch[:, 3] == 6) and
                    np.all(patch[1:3, 1:3] == center)):
                result.append((x, y, center))
    return result


def _find_portals(a, buttons, player):
    big = set()
    small = set()
    indicated = None

    # Six-by-six outer indicator for the next large teleport destination.
    for y in range(59):
        for x in range(59):
            if _inside_button(x, y, buttons):
                continue
            if (a[y, x] == 11 and a[y, x+5] == 11 and
                    a[y+5, x] == 11 and a[y+5, x+5] == 11):
                ix, iy = x + 1, y + 1
                inner = (
                    a[iy, ix] == 11 and a[iy, ix+3] == 11 and
                    a[iy+3, ix] == 11 and a[iy+3, ix+3] == 11
                )
                covered = (player is not None and player[2] == 2 and
                           player[0] == ix and player[1] == iy)
                if inner or covered:
                    big.add((ix, iy))
                    indicated = (ix, iy)

    for y in range(61):
        for x in range(61):
            if _inside_button(x, y, buttons):
                continue
            if not (a[y, x] == 11 and a[y, x+3] == 11 and
                    a[y+3, x] == 11 and a[y+3, x+3] == 11):
                continue
            middle = a[y+1:y+3, x+1:x+3]
            is_small = bool(np.all(middle == 15))
            if player is not None:
                if (player[2] == 1 and player[0] == x + 1 and
                        player[1] == y + 1):
                    is_small = True
            if is_small:
                small.add((x + 1, y + 1))
            else:
                big.add((x, y))

    # Small destinations may be visible without their four corner markers.
    for y in range(63):
        for x in range(63):
            if _inside_button(x, y, buttons):
                continue
            if np.all(a[y:y+2, x:x+2] == 15):
                small.add((x, y))

    return big, small, indicated


def _panel_mask(a):
    mask = 0
    for r in range(3):
        for c in range(3):
            if a[50 + 5*r, 25 + 5*c] == 14:
                mask |= 1 << (3*r + c)
    return mask


def _click_sequence(start, wanted, abilities):
    """Shortest panel sequence that does not accidentally cast another spell."""
    target = SPELLS[wanted]
    triggering = {SPELLS[s] for s in abilities}
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        mask, path = queue.popleft()
        for bit in range(9):
            nxt = mask ^ (1 << bit)
            newpath = path + [bit]
            if nxt == target:
                return newpath
            if nxt in triggering or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, newpath))
    return None


def _build_world(a, goal, player, buttons, switches, big, small):
    # Static collision pixels.
    wall = (a == 5).copy()
    wall |= (a == 4)

    # The action bar and the invisible panel hitboxes are collidable.
    wall[:, 62:64] = True
    wall[47:64, 22:39] = True
    for x, y in buttons:
        wall[y:y+10, x:x+10] = True

    gx, gy, gw, gh = goal
    wall[gy:gy+gh, gx:gx+gw] = True

    # The small teleporter's four blue corners are a collidable sprite.
    for px, py in small:
        x, y = px - 1, py - 1
        if 0 <= x <= 60 and 0 <= y <= 60:
            for xx, yy in ((x, y), (x+3, y), (x, y+3), (x+3, y+3)):
                if a[yy, xx] == 11:
                    wall[yy, xx] = True

    switch_cells = np.full((64, 64), -1, dtype=np.int16)
    for i, (x, y, color) in enumerate(switches):
        switch_cells[y:y+4, x:x+4] = i

    normal_doors = []
    other_doors = []
    for y in range(64):
        for x in range(64):
            if switch_cells[y, x] >= 0:
                continue
            if _inside_button(x, y, buttons):
                continue
            if 22 <= x < 39 and 47 <= y < 64:
                continue
            if a[y, x] == 13:
                normal_doors.append((x, y))
            elif a[y, x] == 12:
                other_doors.append((x, y))

    # The projectile ignores most sprites, including decorative color-4
    # brackets and the interface. It does collide with the color-5 maze.
    fire_wall = (a == 5).copy()

    return wall, fire_wall, switch_cells, normal_doors, other_doors


def _plan(a, goal, player, buttons, switches, abilities,
          big, small, big_index, small_index, initial_mask,
          failmove, failcast, signature):
    wall, fire_wall, switch_cells, normal_doors, other_doors = _build_world(
        a, goal, player, buttons, switches, big, small
    )
    gx, gy, gw, gh = goal
    ns = len(switches)
    if ns > 10:
        switches = switches[:10]
        ns = 10

    normal_bits = 0
    other_bits = 0
    for i, (_, _, color) in enumerate(switches):
        if color == 13:
            normal_bits |= 1 << i
        else:
            other_bits |= 1 << i

    initial_sequences = {}
    empty_sequences = {}
    for spell in abilities:
        initial_sequences[spell] = _click_sequence(initial_mask, spell, abilities)
        empty_sequences[spell] = _click_sequence(0, spell, abilities)

    collision_cache = {}
    resize_cache = {}

    def collision_map(bits):
        if bits in collision_cache:
            return collision_cache[bits]
        arr = wall.copy()
        for i, (x, y, _) in enumerate(switches):
            if not (bits & (1 << i)):
                arr[y:y+4, x:x+4] = True
        if not (bits & normal_bits):
            for x, y in normal_doors:
                arr[y, x] = True
        if not (bits & other_bits):
            for x, y in other_doors:
                arr[y, x] = True
        collision_cache[bits] = arr
        return arr

    def blocked(x, y, size, bits):
        if x < 0 or y < 0 or x + size > 64 or y + size > 64:
            return True
        return bool(np.any(collision_map(bits)[y:y+size, x:x+size]))

    def overlaps_goal(x, y, size):
        return x < gx + gw and x + size > gx and y < gy + gh and y + size > gy

    def grow_position(x, y, bits):
        key = (x, y, bits)
        if key in resize_cache:
            return resize_cache[key]
        # Growth checks maze and doors, but not switches, goal, or UI.
        for ox in range(-2, 1):
            for oy in range(-2, 1):
                nx, ny = x + ox, y + oy
                if nx < 0 or ny < 0 or nx + 4 > 64 or ny + 4 > 64:
                    continue
                ok = True
                for yy in range(ny, ny + 4):
                    for xx in range(nx, nx + 4):
                        if fire_wall[yy, xx]:
                            ok = False
                            break
                        if (a[yy, xx] == 13 and switch_cells[yy, xx] < 0
                                and not (bits & normal_bits)):
                            ok = False
                            break
                        if (a[yy, xx] == 12 and switch_cells[yy, xx] < 0
                                and not (bits & other_bits)):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    resize_cache[key] = (nx, ny)
                    return nx, ny
        resize_cache[key] = None
        return None

    def fire_hit(x, y, scale, direction, bits):
        dx, dy = DIRS[direction]
        size = 2 * scale
        sx = x + (size - 1 if dx > 0 else 0)
        sy = y + (size - 1 if dy > 0 else 0)
        for distance in range(1, 64):
            xx = sx + dx * distance
            yy = sy + dy * distance
            if not (0 <= xx < 64 and 0 <= yy < 64):
                break
            si = int(switch_cells[yy, xx])
            if 0 <= si < ns and not (bits & (1 << si)):
                return si
            if fire_wall[yy, xx]:
                break
            if a[yy, xx] == 13 and si < 0 and not (bits & normal_bits):
                break
            if a[yy, xx] == 12 and si < 0 and not (bits & other_bits):
                break
        return None

    def heuristic(x, y, scale):
        size = 2 * scale
        dx = max(0, gx - (x + size), x - (gx + gw))
        dy = max(0, gy - (y + size), y - (gy + gh))
        h = (dx + dy) // 4
        if "teleport" in abilities:
            for tx, ty in big + small:
                ddx = max(0, gx - (tx + 4), tx - (gx + gw))
                ddy = max(0, gy - (ty + 4), ty - (gy + gh))
                h = min(h, (ddx + ddy) // 4)
        return h

    x, y, scale, direction = player
    use_initial = 1 if initial_mask else 0
    start = (x, y, scale, direction, 0, use_initial, big_index, small_index)
    dist = {start: 0}
    parent = {}
    serial = 0
    heap = [(heuristic(x, y, scale), 0, serial, start)]
    deadline = time.monotonic() + 7.5
    expanded = 0
    found_state = None
    found_event = None

    def push(old, new, cost, event):
        nonlocal serial
        nd = dist[old] + cost
        if nd < dist.get(new, 10**9):
            dist[new] = nd
            parent[new] = (old, event)
            serial += 1
            nx, ny, sc = new[0], new[1], new[2]
            heapq.heappush(heap, (nd + heuristic(nx, ny, sc), nd, serial, new))

    while heap:
        _, cost, _, state = heapq.heappop(heap)
        if cost != dist.get(state):
            continue
        expanded += 1
        if expanded % 512 == 0 and time.monotonic() > deadline:
            break
        if expanded > 150000:
            break

        x, y, scale, direction, bits, ui, bi, si = state
        size = 2 * scale

        for d, (dx, dy) in enumerate(DIRS):
            if bits == 0 and (x, y, scale, d, signature) in failmove:
                continue
            step = size
            nx, ny = x + dx * step, y + dy * step
            if blocked(nx, ny, size, bits):
                if scale == 2:
                    nx, ny = x + dx * 2, y + dy * 2
                    if blocked(nx, ny, size, bits):
                        if overlaps_goal(nx, ny, size):
                            found_state = state
                            found_event = ("win", d, None)
                            break
                        nx, ny = x, y
                else:
                    if overlaps_goal(nx, ny, size):
                        found_state = state
                        found_event = ("win", d, None)
                        break
                    nx, ny = x, y
            new = (nx, ny, scale, d, bits, ui, bi, si)
            if new != state:
                push(state, new, 1, ("move", d, new))
        if found_event is not None:
            break

        sequences = initial_sequences if ui else empty_sequences

        if "resize" in abilities and sequences.get("resize") is not None:
            failkey = ("resize", x, y, scale, direction, signature)
            if not (bits == 0 and failkey in failcast):
                if scale == 2:
                    pos = (x, y)
                    newscale = 1
                else:
                    pos = grow_position(x, y, bits)
                    newscale = 2
                if pos is not None:
                    nx, ny = pos
                    new = (nx, ny, newscale, direction, bits, 0, bi, si)
                    if new != state:
                        push(state, new, len(sequences["resize"]),
                             ("cast", "resize", new, None))

        if "fire" in abilities and sequences.get("fire") is not None and ns:
            failkey = ("fire", x, y, scale, direction, signature)
            if not (bits == 0 and failkey in failcast):
                hit = fire_hit(x, y, scale, direction, bits)
                if hit is not None:
                    new = (x, y, scale, direction, bits | (1 << hit), 0, bi, si)
                    push(state, new, len(sequences["fire"]),
                         ("cast", "fire", new, switches[hit][:2]))

        if "teleport" in abilities and sequences.get("teleport") is not None:
            failkey = ("teleport", x, y, scale, direction, signature)
            if not (bits == 0 and failkey in failcast):
                targets = small if scale == 1 else big
                index = si if scale == 1 else bi
                if targets:
                    index %= len(targets)
                    nx, ny = targets[index]
                    nbi, nsi = bi, si
                    if scale == 1:
                        nsi = (index + 1) % len(targets)
                    else:
                        nbi = (index + 1) % len(targets)
                    new = (nx, ny, scale, direction, bits, 0, nbi, nsi)
                    if new != state:
                        push(state, new, len(sequences["teleport"]),
                             ("cast", "teleport", new, None))

    if found_event is None:
        return []

    events = [found_event]
    cur = found_state
    while cur != start:
        previous, event = parent[cur]
        events.append(event)
        cur = previous
    events.reverse()

    # Only cache the portion through the first spell; then reparse the world.
    plan = []
    current = start
    mask = initial_mask
    for event in events:
        kind = event[0]
        if kind in ("move", "win"):
            d = event[1]
            nxt = event[2]
            record = {
                "action": DIR_ACTION[d],
                "kind": kind,
                "before": list(current[:4]),
                "direction": d,
                "signature": signature,
            }
            if nxt is not None:
                record["expected"] = list(nxt[:4])
                current = nxt
            plan.append(record)
            if kind == "win":
                break
        else:
            _, spell, nxt, hit = event
            seq = _click_sequence(mask, spell, abilities)
            if not seq:
                break
            for j, bit in enumerate(seq):
                mask ^= 1 << bit
                last = j == len(seq) - 1
                record = {
                    "action": "ACTION6",
                    "row": 50 + 5 * (bit // 3),
                    "col": 25 + 5 * (bit % 3),
                    "kind": "cast" if last else "click",
                    "mask": 0 if last else mask,
                    "signature": signature,
                }
                if last:
                    record["spell"] = spell
                    record["before"] = list(current[:4])
                    record["expected"] = list(nxt[:4])
                    record["hit"] = hit
                    record["next_big"] = nxt[6]
                    record["next_small"] = nxt[7]
                plan.append(record)
            break
    return plan


def _emit(record, memory):
    memory["pending"] = record
    if record["action"] == "ACTION6":
        return "ACTION6", int(record["row"]), int(record["col"])
    return record["action"], None, None


def act(obs, available, memory):
    a = np.asarray(obs)
    buttons, abilities = _find_buttons(a)
    goal, player = _find_goal_and_player(a)

    if goal is None or player is None:
        # A reset is preferable to blindly walking when the player cannot be
        # identified. Avoid repeated resets if the observation remains unclear.
        if not memory.get("parse_reset") and "RESET" in available:
            memory["parse_reset"] = True
            memory["plan"] = []
            memory["pending"] = None
            return "RESET", None, None
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if name in available:
                return name, None, None
        return available[0], None, None

    memory["parse_reset"] = False
    abilities |= set(memory.get("abilities", []))
    memory["abilities"] = sorted(abilities)
    switches = _find_switches(a, buttons)
    big_now, small_now, indicated = _find_portals(a, buttons, player)

    big = sorted(set(tuple(p) for p in memory.get("big", [])) | big_now)
    small = sorted(set(tuple(p) for p in memory.get("small", [])) | small_now)
    memory["big"] = [list(p) for p in big]
    memory["small"] = [list(p) for p in small]

    if indicated in big:
        memory["big_index"] = big.index(indicated)
    big_index = memory.get("big_index", 0) % max(1, len(big))
    small_index = memory.get("small_index", 0) % max(1, len(small))
    mask = _panel_mask(a)

    # This signature distinguishes failures before and after doors change.
    signature = (
        tuple(switches),
        int(np.count_nonzero(a == 13)),
        int(np.count_nonzero(a == 12)),
    )
    failmove = memory.setdefault("failmove", set())
    failcast = memory.setdefault("failcast", set())

    pending = memory.pop("pending", None)
    if pending:
        kind = pending.get("kind")
        ok = True
        if kind == "win":
            ok = False  # If called again, the predicted win did not happen.
        elif kind in ("move", "cast"):
            expected = pending.get("expected")
            if expected is not None and tuple(expected) != tuple(player):
                ok = False

        if kind in ("click", "cast") and mask != pending.get("mask"):
            ok = False

        if kind == "cast" and pending.get("spell") == "fire":
            hit = pending.get("hit")
            if hit is not None and any((x, y) == tuple(hit) for x, y, _ in switches):
                ok = False

        if not ok:
            memory["plan"] = []
            before = pending.get("before")
            if kind in ("move", "win") and before is not None:
                expected = pending.get("expected")
                expected_movement = (
                    kind == "win" or
                    (expected is not None and tuple(expected[:2]) != tuple(before[:2]))
                )
                if expected_movement and tuple(player[:2]) == tuple(before[:2]):
                    failmove.add((
                        before[0], before[1], before[2],
                        pending["direction"], pending["signature"]
                    ))
            elif kind == "cast" and before is not None:
                failcast.add((
                    pending["spell"], before[0], before[1], before[2],
                    before[3], pending["signature"]
                ))
        elif kind == "cast":
            memory["big_index"] = pending.get("next_big", big_index)
            memory["small_index"] = pending.get("next_small", small_index)
            # The outer indicator is authoritative when it is visible.
            if indicated in big:
                memory["big_index"] = big.index(indicated)
            memory["plan"] = []
            big_index = memory["big_index"]
            small_index = memory["small_index"]

    plan = memory.get("plan", [])
    if not plan:
        plan = _plan(
            a, goal, player, buttons, switches, abilities,
            big, small, big_index, small_index, mask,
            failmove, failcast, signature,
        )
        memory["plan"] = plan

    if plan:
        record = plan.pop(0)
        memory["plan"] = plan
        if record["action"] in available:
            return _emit(record, memory)

    # If no modelled route exists, clear a partial panel or explore.
    if mask and "ACTION6" in available:
        bit = next(i for i in range(9) if mask & (1 << i))
        record = {
            "action": "ACTION6",
            "row": 50 + 5 * (bit // 3),
            "col": 25 + 5 * (bit % 3),
            "kind": "click",
            "mask": mask ^ (1 << bit),
            "signature": signature,
        }
        return _emit(record, memory)

    if not memory.get("route_reset") and "RESET" in available:
        memory["route_reset"] = True
        memory["plan"] = []
        memory["pending"] = None
        memory["failmove"] = set()
        memory["failcast"] = set()
        memory["big_index"] = 0
        memory["small_index"] = 0
        return "RESET", None, None

    # Deterministic fallback, biased toward the exit.
    x, y, scale, direction = player
    gx, gy, gw, gh = goal
    choices = sorted(
        range(4),
        key=lambda d: abs(x + DIRS[d][0] * 2 * scale - gx)
                    + abs(y + DIRS[d][1] * 2 * scale - gy),
    )
    counter = memory.get("fallback_count", 0)
    memory["fallback_count"] = counter + 1
    d = choices[counter % len(choices)]
    action = DIR_ACTION[d]
    if action not in available:
        action = next((v for v in DIR_ACTION if v in available), available[0])
    return action, None, None

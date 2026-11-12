import heapq
import math
import time
from collections import deque

import numpy as np


# Spell cells, in row-major order.
SPELLS = {
    "fire": (1 << 1) | (1 << 4) | (1 << 7),
    "warp": (1 << 0) | (1 << 1) | (1 << 4),
    "size": (1 << 1) | (1 << 3) | (1 << 5) | (1 << 7),
}
CELL_X = (25, 30, 35)
CELL_Y = (50, 55, 60)


def _rotations(a):
    return [np.rot90(a, -k) for k in range(4)]


def _matches(obs, template):
    h, w = template.shape
    result = []
    for y in range(65 - h):
        for x in range(65 - w):
            if np.array_equal(obs[y:y + h, x:x + w], template):
                result.append((x, y, w, h))
    return result


def _find_door(obs):
    base = np.array([
        [9, 9, 9, 9, 9],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 10, 10, 10, 10],
        [9, 9, 9, 9, 9],
    ])
    found = []
    for t in _rotations(base):
        found.extend(_matches(obs, t))
    if found:
        return found[0]

    # Fallback for a partially obscured door.
    seen = set()
    components = []
    for y in range(64):
        for x in range(64):
            if (x, y) in seen or int(obs[y, x]) not in (9, 10):
                continue
            q = [(x, y)]
            seen.add((x, y))
            comp = []
            while q:
                p = q.pop()
                comp.append(p)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = p[0] + dx, p[1] + dy
                    if (0 <= nx < 64 and 0 <= ny < 64
                            and (nx, ny) not in seen
                            and int(obs[ny, nx]) in (9, 10)):
                        seen.add((nx, ny))
                        q.append((nx, ny))
            if len(comp) >= 20:
                xs = [p[0] for p in comp]
                ys = [p[1] for p in comp]
                components.append((min(xs), min(ys),
                                   max(xs) - min(xs) + 1,
                                   max(ys) - min(ys) + 1))
    return components[0] if components else None


def _find_player(obs, door):
    base = np.array([[10, 10], [9, 9]])
    # Rotation order corresponds to down, left, up, right.
    directions = (1, 2, 0, 3)
    for scale in (2, 1):
        for k, t in enumerate(_rotations(base)):
            t = np.repeat(np.repeat(t, scale, axis=0), scale, axis=1)
            for x, y, w, h in _matches(obs, t):
                if door:
                    gx, gy, gw, gh = door
                    if x < gx + gw and x + w > gx and y < gy + gh and y + h > gy:
                        continue
                return x, y, scale, directions[k]
    return None


def _icons(obs):
    icons = []
    for y in range(55):
        for x in range(55):
            if (np.all(obs[y, x:x + 10] == 3)
                    and np.all(obs[y + 9, x:x + 10] == 3)
                    and np.all(obs[y:y + 10, x] == 3)
                    and np.all(obs[y:y + 10, x + 9] == 3)):
                inner = obs[y + 1:y + 9, x + 1:x + 9]
                colors = set(int(v) for v in np.unique(inner))
                spells = set()
                if 6 in colors:
                    spells.add("fire")
                if 11 in colors:
                    spells.add("warp")
                if 15 in colors:
                    spells.add("size")
                if spells:
                    icons.append((x, y, spells))
    return icons


def _in_ui(x, y, icons):
    if 22 <= x < 39 and 47 <= y < 64:
        return True
    for ix, iy, _ in icons:
        if ix <= x < ix + 10 and iy <= y < iy + 10:
            return True
    return False


def _read_grid(obs):
    mask = 0
    for r in range(3):
        for c in range(3):
            if int(obs[CELL_Y[r], CELL_X[c]]) == 14:
                mask |= 1 << (3 * r + c)
    return mask


def _objects(obs, icons):
    targets = []
    # Fire targets have a 6-colored border and a 2x2 center.
    for y in range(61):
        for x in range(61):
            if _in_ui(x, y, icons):
                continue
            a = obs[y:y + 4, x:x + 4]
            if not (np.all(a[0, :] == 6) and np.all(a[3, :] == 6)
                    and np.all(a[:, 0] == 6) and np.all(a[:, 3] == 6)):
                continue
            center = int(a[1, 1])
            if center in (12, 13) and np.all(a[1:3, 1:3] == center):
                targets.append((x, y, 0 if center == 13 else 1))

    gates = [[], []]
    for group, color in enumerate((13, 12)):
        seen = set()
        for y in range(64):
            for x in range(64):
                if (x, y) in seen or int(obs[y, x]) != color:
                    continue
                q = [(x, y)]
                seen.add((x, y))
                comp = []
                while q:
                    px, py = q.pop()
                    comp.append((px, py))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = px + dx, py + dy
                        if (0 <= nx < 64 and 0 <= ny < 64
                                and (nx, ny) not in seen
                                and int(obs[ny, nx]) == color):
                            seen.add((nx, ny))
                            q.append((nx, ny))
                if len(comp) >= 6 and not _in_ui(x, y, icons):
                    gates[group].extend(comp)

    small = []
    for y in range(63):
        for x in range(63):
            if not _in_ui(x, y, icons) and np.all(obs[y:y + 2, x:x + 2] == 15):
                small.append((x, y))

    large = []
    small_rings = []
    for y in range(61):
        for x in range(61):
            if _in_ui(x, y, icons):
                continue
            if all(int(obs[yy, xx]) == 11 for xx, yy in
                   ((x, y), (x + 3, y), (x, y + 3), (x + 3, y + 3))):
                if any(x < sx < x + 3 and y < sy < y + 3 for sx, sy in small):
                    small_rings.append((x, y))
                else:
                    large.append((x, y))

    return targets, gates, large, small, small_rings


def _click_sequence(start, spell, allowed):
    """Shortest toggling sequence that does not cast a different spell."""
    target = SPELLS[spell]
    q = deque([(start, [])])
    seen = {start}
    while q:
        mask, path = q.popleft()
        for i in range(9):
            new = mask ^ (1 << i)
            if new == target:
                return path + [i]
            if new in seen:
                continue
            if any(new == SPELLS[s] for s in allowed):
                continue
            seen.add(new)
            q.append((new, path + [i]))
    return None


def _rows_from_cells(cells):
    rows = [0] * 64
    for x, y in cells:
        if 0 <= x < 64 and 0 <= y < 64:
            rows[y] |= 1 << x
    return rows


def _plan(obs, player, door, memory):
    started = time.monotonic()
    icons = _icons(obs)
    allowed = set()
    for _, _, names in icons:
        allowed.update(names)

    targets, gates, large, small, small_rings = _objects(obs, icons)
    if not allowed:
        if targets:
            allowed.add("fire")
        if large or small:
            allowed.add("warp")

    px, py, scale, direction = player
    size = 2 * scale
    gx, gy, gw, gh = door
    grid0 = _read_grid(obs)

    # Static collision pixels. The 3x3 spell interface is collidable as well.
    walls = [0] * 64
    ray_walls = [0] * 64
    for y in range(64):
        for x in range(64):
            v = int(obs[y, x])
            if v == 5:
                ray_walls[y] |= 1 << x
            if v in (1, 3, 4, 5) or (v == 14 and x >= 62):
                walls[y] |= 1 << x

    if int(obs[47, 22]) == 3:
        for y in range(47, 64):
            walls[y] |= ((1 << 17) - 1) << 22

    for ix, iy, _ in icons:
        for y in range(iy, min(64, iy + 10)):
            walls[y] |= ((1 << min(10, 64 - ix)) - 1) << ix

    # The small teleport marker is surrounded by a collidable corner sprite.
    for x, y in small_rings:
        for xx, yy in ((x, y), (x + 3, y), (x, y + 3), (x + 3, y + 3)):
            walls[yy] |= 1 << xx

    # The player and door cover the underlying terrain.
    for y in range(py, min(64, py + size)):
        mask = ((1 << min(size, 64 - px)) - 1) << px
        walls[y] &= ~mask
        ray_walls[y] &= ~mask

    door_rows = [0] * 64
    for y in range(gy, min(64, gy + gh)):
        mask = ((1 << min(gw, 64 - gx)) - 1) << gx
        door_rows[y] |= mask
        walls[y] &= ~mask
        ray_walls[y] &= ~mask

    gate_rows = [_rows_from_cells(g) for g in gates]
    target_rows = [[0] * 64 for _ in range(2)]
    for x, y, group in targets:
        for yy in range(y, y + 4):
            target_rows[group][yy] |= 15 << x

    # A target's rendered pixels cover any guessed static terrain.
    for x, y, group in targets:
        for yy in range(y, y + 4):
            mask = 15 << x
            walls[yy] &= ~mask
            ray_walls[yy] &= ~mask

    forbidden = {tuple(v) for v in memory.get("blocked", [])}

    def intersects(rows, x, y, n):
        if x < 0 or y < 0 or x + n > 64 or y + n > 64:
            return True
        bits = ((1 << n) - 1) << x
        return any(rows[yy] & bits for yy in range(y, y + n))

    def occupied(x, y, n, removed, include_targets=True):
        if (x, y, n) in forbidden:
            return True
        if intersects(walls, x, y, n):
            return True
        for group in range(2):
            if not (removed & (1 << group)):
                if intersects(gate_rows[group], x, y, n):
                    return True
                if include_targets and intersects(target_rows[group], x, y, n):
                    return True
        return False

    def growth_clear(x, y, removed):
        # Growth checks walls and gates, but not fire targets or decorations.
        if x < 0 or y < 0 or x + 4 > 64 or y + 4 > 64:
            return False
        if intersects(ray_walls, x, y, 4):
            return False
        for group in range(2):
            if not (removed & (1 << group)):
                if intersects(gate_rows[group], x, y, 4):
                    return False
        return True

    def shot(x, y, n, d, removed):
        dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[d]
        sx = x + (n - 1 if dx > 0 else 0)
        sy = y + (n - 1 if dy > 0 else 0)
        for distance in range(1, 64):
            xx, yy = sx + dx * distance, sy + dy * distance
            if not (0 <= xx < 64 and 0 <= yy < 64):
                break
            bit = 1 << xx
            for group in range(2):
                if not (removed & (1 << group)):
                    if target_rows[group][yy] & bit:
                        return group
                    if gate_rows[group][yy] & bit:
                        return None
            if ray_walls[yy] & bit:
                return None
        return None

    click_cache = {}
    for mask in {grid0, 0}:
        for spell in allowed:
            click_cache[(mask, spell)] = _click_sequence(mask, spell, allowed)

    def distance_to_goal(x, y, n):
        dx = max(gx - (x + n - 1), x - (gx + gw - 1), 0)
        dy = max(gy - (y + n - 1), y - (gy + gh - 1), 0)
        return (dx + dy + 3) // 4

    def heuristic(x, y, n):
        h = distance_to_goal(x, y, n)
        if "warp" in allowed:
            for tx, ty in large:
                h = min(h, 4 + distance_to_goal(tx, ty, 4))
            for tx, ty in small:
                h = min(h, 4 + distance_to_goal(tx, ty, 2))
        return h

    iw = int(memory.get("large_index", 0)) % max(1, len(large))
    iv = int(memory.get("small_index", 0)) % max(1, len(small))
    # State: x, y, scale, facing, removed groups, portal indices, cast-used.
    start = (px, py, scale, direction, 0, iw, iv, 0)
    dist = {start: 0}
    parent = {}
    heap = [(heuristic(px, py, size), 0, 0, start)]
    serial = 0
    best = start
    best_h = heuristic(px, py, size)
    goal_state = None
    goal_edge = None
    expanded = 0

    def push(old, new, cost, edge):
        nonlocal serial, best, best_h
        ng = dist[old] + cost
        if ng >= dist.get(new, 10**9):
            return
        dist[new] = ng
        parent[new] = (old, edge)
        h = heuristic(new[0], new[1], 2 * new[2])
        if h < best_h:
            best_h = h
            best = new
        serial += 1
        heapq.heappush(heap, (ng + h, ng, serial, new))

    while heap and expanded < 120000:
        if expanded % 1024 == 0 and time.monotonic() - started > 7.0:
            break
        _, cost, _, state = heapq.heappop(heap)
        if cost != dist.get(state):
            continue
        expanded += 1
        x, y, sc, d, removed, li, si, used = state
        n = 2 * sc

        for action, (dx, dy) in enumerate(
                ((0, -1), (0, 1), (-1, 0), (1, 0)), 1):
            step = n
            nx, ny = x + dx * step, y + dy * step
            blocked = occupied(nx, ny, n, removed)
            door_hit = intersects(door_rows, nx, ny, n) if (
                0 <= nx <= 64 - n and 0 <= ny <= 64 - n) else False
            blocked = blocked or door_hit

            if blocked and sc == 2:
                nx, ny = x + dx * 2, y + dy * 2
                blocked = occupied(nx, ny, n, removed)
                door_hit = intersects(door_rows, nx, ny, n) if (
                    0 <= nx <= 64 - n and 0 <= ny <= 64 - n) else False
                blocked = blocked or door_hit

            if blocked:
                if door_hit:
                    goal_state = state
                    goal_edge = ("move", action, x, y, sc)
                    break
                # A blocked movement can still change facing, useful for firing.
                new = (x, y, sc, action - 1, removed, li, si, used)
                if new != state:
                    push(state, new, 1, ("move", action, x, y, sc))
            else:
                new = (nx, ny, sc, action - 1, removed, li, si, used)
                push(state, new, 1, ("move", action, nx, ny, sc))
        if goal_state is not None:
            break

        current_mask = 0 if used else grid0

        if "fire" in allowed:
            group = shot(x, y, n, d, removed)
            clicks = click_cache.get((current_mask, "fire"))
            if group is not None and clicks:
                new = (x, y, sc, d, removed | (1 << group), li, si, 1)
                push(state, new, len(clicks) + 1,
                     ("cast", "fire", clicks, x, y, sc, group))

        if "size" in allowed:
            clicks = click_cache.get((current_mask, "size"))
            if clicks:
                if sc == 2:
                    new = (x, y, 1, d, removed, li, si, 1)
                    push(state, new, len(clicks) + 1,
                         ("cast", "size", clicks, x, y, 1, -1))
                else:
                    offset = None
                    for ox in (-2, -1, 0):
                        for oy in (-2, -1, 0):
                            if growth_clear(x + ox, y + oy, removed):
                                offset = (ox, oy)
                                break
                        if offset is not None:
                            break
                    if offset is not None:
                        nx, ny = x + offset[0], y + offset[1]
                        new = (nx, ny, 2, d, removed, li, si, 1)
                        push(state, new, len(clicks) + 1,
                             ("cast", "size", clicks, nx, ny, 2, -1))

        if "warp" in allowed:
            clicks = click_cache.get((current_mask, "warp"))
            portals = large if sc == 2 else small
            if clicks and portals:
                index = li if sc == 2 else si
                tx, ty = portals[index % len(portals)]
                nli, nsi = li, si
                if sc == 2:
                    nli = (li + 1) % len(portals)
                else:
                    nsi = (si + 1) % len(portals)
                new = (tx, ty, sc, d, removed, nli, nsi, 1)
                if new != state:
                    push(state, new, len(clicks) + 1,
                         ("cast", "warp", clicks, tx, ty, sc, -1))

    end = goal_state if goal_state is not None else best
    edges = []
    while end != start and end in parent:
        previous, edge = parent[end]
        edges.append(edge)
        end = previous
    edges.reverse()
    if goal_edge is not None:
        edges.append(goal_edge)
    if not edges:
        return []

    queue = []
    mask = grid0
    cx, cy, cs = px, py, scale
    for edge in edges:
        if edge[0] == "move":
            _, action, nx, ny, ns = edge
            queue.append({
                "action": "ACTION%d" % action,
                "expected": [nx, ny, ns, mask],
                "kind": "move",
            })
            cx, cy, cs = nx, ny, ns
        else:
            _, spell, clicks, nx, ny, ns, group = edge
            for j, cell in enumerate(clicks):
                mask ^= 1 << cell
                last = j == len(clicks) - 1
                if last:
                    mask = 0
                    ex, ey, es = nx, ny, ns
                else:
                    ex, ey, es = cx, cy, cs
                item = {
                    "action": "ACTION6",
                    "row": CELL_Y[cell // 3],
                    "col": CELL_X[cell % 3],
                    "expected": [ex, ey, es, mask],
                    "kind": spell if last else "click",
                }
                if last and spell == "warp":
                    item["portal_scale"] = cs
                if last and spell == "fire":
                    item["fire_group"] = group
                queue.append(item)
            cx, cy, cs = nx, ny, ns
    return queue


def act(obs, available, memory):
    obs = np.asarray(obs)
    door = _find_door(obs)
    player = _find_player(obs, door) if door else None
    mask = _read_grid(obs)

    previous = memory.pop("pending", None)
    if previous is not None and player is not None:
        actual = [player[0], player[1], player[2], mask]
        expected = previous.get("expected")
        if expected != actual:
            # Learn unexpected blocked destinations, except on the very first
            # action, which can be consumed by the level-one tutorial.
            if (previous.get("kind") == "move"
                    and memory.get("issued", 0) > 1
                    and expected
                    and expected[:2] != actual[:2]):
                blocked = memory.setdefault("blocked", [])
                entry = [expected[0], expected[1], 2 * expected[2]]
                if entry not in blocked:
                    blocked.append(entry)
            memory["queue"] = []

    if player is None or door is None:
        # Do not repeatedly reset on a transient or unrecognized frame.
        for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if action in available:
                return action, None, None
        return "RESET", None, None

    queue = memory.get("queue", [])
    if not queue:
        queue = _plan(obs, player, door, memory)
        memory["queue"] = queue

    if not queue:
        if "RESET" in available and memory.get("issued", 0) > 0:
            memory.clear()
            return "RESET", None, None
        return "ACTION1", None, None

    item = queue.pop(0)
    memory["queue"] = queue
    memory["pending"] = item
    memory["issued"] = memory.get("issued", 0) + 1

    if item.get("kind") == "warp":
        if item.get("portal_scale") == 2:
            memory["large_index"] = memory.get("large_index", 0) + 1
        else:
            memory["small_index"] = memory.get("small_index", 0) + 1

    action = item["action"]
    if action not in available:
        memory["queue"] = []
        for candidate in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if candidate in available:
                return candidate, None, None
        return "RESET", None, None

    if action == "ACTION6":
        return action, int(item["row"]), int(item["col"])
    return action, None, None

import heapq
import itertools
from collections import deque

import numpy as np


# Spell patterns, as bit positions in the 3-by-3 panel.
SPELLS = {
    "fire": (1 << 1) | (1 << 4) | (1 << 7),
    "teleport": (1 << 0) | (1 << 1) | (1 << 4),
    "resize": (1 << 1) | (1 << 3) | (1 << 5) | (1 << 7),
}


def _matches(obs, pattern):
    """Return top-left positions where a small pattern matches exactly."""
    h, w = pattern.shape
    ok = np.ones((65 - h, 65 - w), dtype=bool)
    for y in range(h):
        for x in range(w):
            ok &= obs[y:y + 65 - h, x:x + 65 - w] == pattern[y, x]
    yy, xx = np.nonzero(ok)
    return list(zip(xx.tolist(), yy.tolist()))


def _find_goal_and_player(obs):
    goal_base = np.array(
        [[9, 9, 9, 9, 9],
         [9, 10, 10, 10, 10],
         [9, 10, 10, 10, 10],
         [9, 10, 10, 10, 10],
         [9, 10, 10, 10, 10],
         [9, 9, 9, 9, 9]],
        dtype=np.int8,
    )
    goals = []
    for k in range(4):
        p = np.rot90(goal_base, -k)
        for x, y in _matches(obs, p):
            goals.append((x, y, p.shape[1], p.shape[0]))
    if not goals:
        return None, None
    goal = goals[0]
    gx, gy, gw, gh = goal

    base = np.array([[10, 10], [9, 9]], dtype=np.int8)
    # Clockwise rotations of the sprite correspond to down, left, up, right.
    directions = (1, 2, 0, 3)
    candidates = []
    for scale in (2, 1):
        p0 = np.repeat(np.repeat(base, scale, axis=0), scale, axis=1)
        for k in range(4):
            p = np.rot90(p0, -k)
            for x, y in _matches(obs, p):
                size = 2 * scale
                if x < gx + gw and x + size > gx and y < gy + gh and y + size > gy:
                    continue
                candidates.append((scale, x, y, directions[k]))
        if candidates:
            break
    if not candidates:
        return goal, None
    scale, x, y, direction = candidates[0]
    return goal, (x, y, scale, direction)


def _find_icons(obs):
    icons = []
    for y in range(55):
        for x in range(55):
            if obs[y, x] != 3:
                continue
            if not (
                np.all(obs[y, x:x + 10] == 3)
                and np.all(obs[y + 9, x:x + 10] == 3)
                and np.all(obs[y:y + 10, x] == 3)
                and np.all(obs[y:y + 10, x + 9] == 3)
            ):
                continue
            inside = obs[y + 1:y + 9, x + 1:x + 9]
            colors = set(int(v) for v in inside.flat)
            kinds = []
            if 6 in colors:
                kinds.append("fire")
            if 11 in colors:
                kinds.append("teleport")
            if 15 in colors:
                kinds.append("resize")
            if kinds:
                icons.append((x, y, kinds[0]))
    return icons


def _in_icon(x, y, icons):
    return any(ix <= x < ix + 10 and iy <= y < iy + 10
               for ix, iy, _ in icons)


def _find_targets(obs, icons):
    """Find the two kinds of 4x4 fire targets, allowing player occlusion."""
    found = []
    for y in range(61):
        for x in range(61):
            if _in_icon(x, y, icons) or (22 <= x <= 38 and y >= 47):
                continue
            block = obs[y:y + 4, x:x + 4]
            ring = []
            center = []
            for dy in range(4):
                for dx in range(4):
                    v = int(block[dy, dx])
                    if dx in (0, 3) or dy in (0, 3):
                        ring.append(v)
                    else:
                        center.append(v)
            if sum(v == 6 for v in ring) < 7:
                continue
            if any(v not in (6, 9, 10) for v in ring):
                continue
            for color, group in ((13, 0), (12, 1)):
                if color not in center:
                    continue
                if all(v in (color, 9, 10) for v in center):
                    found.append((x, y, group))
                    break
    return found


def _find_teleports(obs, icons):
    small = []
    for y in range(63):
        for x in range(63):
            if _in_icon(x, y, icons) or (22 <= x <= 38 and y >= 47):
                continue
            if np.all(obs[y:y + 2, x:x + 2] == 15):
                small.append((x, y))

    big = []
    marked = None
    for y in range(61):
        for x in range(61):
            if _in_icon(x, y, icons) or (22 <= x <= 38 and y >= 47):
                continue
            corners = (obs[y, x], obs[y, x + 3],
                       obs[y + 3, x], obs[y + 3, x + 3])
            if not all(int(v) == 11 for v in corners):
                continue
            # A four-corner decoration around a small teleport is not itself
            # a large teleport.
            if np.any(obs[y + 1:y + 3, x + 1:x + 3] == 15):
                continue
            big.append((x, y))
            if 1 <= x <= 59 and 1 <= y <= 59:
                outer = (obs[y - 1, x - 1], obs[y - 1, x + 4],
                         obs[y + 4, x - 1], obs[y + 4, x + 4])
                if all(int(v) == 11 for v in outer):
                    marked = (x, y)

    # The visible outer frame identifies the next large destination.
    if marked in big:
        i = big.index(marked)
        big = big[i:] + big[:i]
    return big, small


def _panel_mask(obs):
    mask = 0
    for r in range(3):
        for c in range(3):
            if int(obs[50 + 5 * r, 25 + 5 * c]) == 14:
                mask |= 1 << (3 * r + c)
    return mask


def _spell_paths(start_mask, allowed):
    """Shortest safe click sequences; another spell must not trigger en route."""
    terminal = {SPELLS[name]: name for name in allowed}
    paths = {}
    q = deque([(start_mask, ())])
    seen = {start_mask}
    while q and len(paths) < len(allowed):
        mask, path = q.popleft()
        for bit in range(9):
            nm = mask ^ (1 << bit)
            if nm in seen:
                continue
            npath = path + (bit,)
            seen.add(nm)
            if nm in terminal:
                paths[terminal[nm]] = npath
            else:
                q.append((nm, npath))
    return paths


def _click(bit):
    return ("ACTION6", 50 + 5 * (bit // 3), 25 + 5 * (bit % 3))


def _make_world(obs, goal, icons, targets, small):
    gx, gy, gw, gh = goal
    wall = [0] * 64
    hard = [0] * 64
    groups = [[0] * 64 for _ in range(2)]
    doors = [[0] * 64 for _ in range(2)]
    goal_rows = [0] * 64

    target_at = {}
    for tx, ty, group in targets:
        for yy in range(ty, ty + 4):
            for xx in range(tx, tx + 4):
                target_at[(xx, yy)] = group

    for y in range(64):
        for x in range(64):
            v = int(obs[y, x])
            bit = 1 << x
            if v == 5:
                wall[y] |= bit
                hard[y] |= bit
            elif v == 4:
                # The decorative brackets are collidable.
                hard[y] |= bit

            group = target_at.get((x, y))
            if group is not None:
                groups[group][y] |= bit
            elif not _in_icon(x, y, icons) and not (22 <= x <= 38 and y >= 47):
                if v == 13:
                    groups[0][y] |= bit
                    doors[0][y] |= bit
                elif v == 12:
                    groups[1][y] |= bit
                    doors[1][y] |= bit

    # The small-teleport frame is a separate, collidable corner sprite.
    for x, y in small:
        for xx, yy in ((x - 1, y - 1), (x + 2, y - 1),
                       (x - 1, y + 2), (x + 2, y + 2)):
            if 0 <= xx < 64 and 0 <= yy < 64 and obs[yy, xx] == 11:
                hard[yy] |= 1 << xx

    # Spell buttons and the full invisible 3x3 click panel are collidable.
    for ix, iy, _ in icons:
        bits = ((1 << 10) - 1) << ix
        for yy in range(iy, min(64, iy + 10)):
            hard[yy] |= bits
            wall[yy] |= bits
    panel_bits = ((1 << 17) - 1) << 22
    for yy in range(47, 64):
        hard[yy] |= panel_bits
        wall[yy] |= panel_bits

    # The action bar is a collidable sprite.
    for yy in range(64):
        hard[yy] |= 3 << 62
        wall[yy] |= 3 << 62

    gbits = ((1 << gw) - 1) << gx
    for yy in range(gy, gy + gh):
        goal_rows[yy] |= gbits

    # Collision lookup: bit 0 means ordinary obstruction, bit 1 means goal.
    tables = {}
    resize_tables = {}
    for flags in range(4):
        rows = []
        growth_rows = []
        for y in range(64):
            r = hard[y]
            gr = wall[y]
            for group in range(2):
                if not (flags & (1 << group)):
                    r |= groups[group][y]
                    gr |= doors[group][y]
            rows.append(r)
            growth_rows.append(gr)

        for scale in (1, 2):
            size = 2 * scale
            table = bytearray(4096)
            for y in range(65 - size):
                combined = 0
                goal_combined = 0
                for yy in range(y, y + size):
                    combined |= rows[yy]
                    goal_combined |= goal_rows[yy]
                width_mask = (1 << size) - 1
                for x in range(65 - size):
                    bits = width_mask << x
                    value = (1 if combined & bits else 0)
                    if goal_combined & bits:
                        value |= 2
                    table[y * 64 + x] = value
            tables[(flags, scale)] = table

        # Growth checks only walls and doors, not targets or the goal.
        growth = bytearray(4096)
        for y in range(61):
            combined = growth_rows[y] | growth_rows[y + 1] | growth_rows[y + 2] | growth_rows[y + 3]
            for x in range(61):
                if combined & (15 << x):
                    growth[y * 64 + x] = 1
        resize_tables[flags] = growth

    return wall, groups, doors, tables, resize_tables


def _search(obs, goal, player, icons, targets, big, small, mask, memory):
    allowed = {kind for _, _, kind in icons}
    if not allowed:
        # This fallback is useful if a button is temporarily obscured.
        allowed = set(SPELLS)
    current_paths = _spell_paths(mask, allowed)
    empty_paths = _spell_paths(0, allowed)

    wall, groups, doors, tables, growth_tables = _make_world(
        obs, goal, icons, targets, small
    )
    gx, gy, gw, gh = goal
    x0, y0, s0, d0 = player

    # Estimate the next small teleport from casts made during this episode.
    small_index = int(memory.get("small_index", 0))
    if small:
        small_index %= len(small)
    else:
        small_index = 0

    # Large destinations were rotated above to put the visible next one first.
    big_index = 0

    def direct_h(x, y):
        dx = max(0, gx - (x + 3), x - (gx + gw - 1))
        dy = max(0, gy - (y + 3), y - (gy + gh - 1))
        return max(1, (dx + dy + 3) // 4)

    tele_h = None
    if "teleport" in allowed and (big or small):
        tele_h = 1 + min(direct_h(x, y) for x, y in big + small)

    def heuristic(x, y):
        h = direct_h(x, y)
        return min(h, tele_h) if tele_h is not None else h

    def collision(flags, scale, x, y):
        size = 2 * scale
        if x < 0 or y < 0 or x + size > 64 or y + size > 64:
            return 1
        return tables[(flags, scale)][y * 64 + x]

    fire_cache = {}

    def fire_hit(x, y, scale, direction, flags):
        key = (x, y, scale, direction, flags)
        if key in fire_cache:
            return fire_cache[key]
        dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[direction]
        size = 2 * scale
        sx = x + (size - 1 if dx > 0 else 0)
        sy = y + (size - 1 if dy > 0 else 0)
        result = None
        for i in range(1, 65):
            xx, yy = sx + dx * i, sy + dy * i
            if not (0 <= xx < 64 and 0 <= yy < 64):
                break
            bit = 1 << xx
            if wall[yy] & bit:
                break
            hit_target = None
            for tx, ty, group in targets:
                if not (flags & (1 << group)) and tx <= xx < tx + 4 and ty <= yy < ty + 4:
                    hit_target = group
                    break
            if hit_target is not None:
                result = hit_target
                break
            if any(not (flags & (1 << group)) and doors[group][yy] & bit
                   for group in range(2)):
                break
        fire_cache[key] = result
        return result

    # State: x, y, scale, direction, opened-target bits, initial-panel-mode,
    # large-teleport index, small-teleport index.
    mode0 = 1 if mask else 0
    start = (x0, y0, s0, d0, 0, mode0, big_index, small_index)
    dist = {start: 0}
    counter = itertools.count()
    heap = [(heuristic(x0, y0), 0, next(counter), start, None)]
    expanded = 0
    max_expanded = 120000

    while heap and expanded < max_expanded:
        f, cost, _, state, first = heapq.heappop(heap)
        if state is None:
            return first
        if cost != dist.get(state):
            continue
        expanded += 1
        x, y, scale, direction, flags, mode, bi, si = state

        def push(ns, extra, action):
            nc = cost + extra
            if nc >= dist.get(ns, 10**9):
                return
            dist[ns] = nc
            fa = first if first is not None else action
            heapq.heappush(
                heap, (nc + heuristic(ns[0], ns[1]), nc,
                       next(counter), ns, fa)
            )

        for nd, (dx, dy) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
            step = 2 if scale == 1 else 4
            nx, ny = x + dx * step, y + dy * step
            hit = collision(flags, scale, nx, ny)
            if hit and scale == 2:
                nx, ny = x + dx * 2, y + dy * 2
                hit = collision(flags, scale, nx, ny)
            action = ("ACTION" + str(nd + 1), None, None)
            if hit & 2:
                fa = first if first is not None else action
                heapq.heappush(heap, (cost + 1, cost + 1,
                                      next(counter), None, fa))
            elif hit:
                if nd != direction:
                    push((x, y, scale, nd, flags, mode, bi, si), 1, action)
            else:
                push((nx, ny, scale, nd, flags, mode, bi, si), 1, action)

        paths = current_paths if mode else empty_paths

        if "resize" in paths:
            path = paths["resize"]
            if path:
                action = _click(path[0])
                if scale == 2:
                    push((x, y, 1, direction, flags, 0, bi, si),
                         len(path), action)
                else:
                    growth = growth_tables[flags]
                    position = None
                    for ox in range(-2, 1):
                        for oy in range(-2, 1):
                            xx, yy = x + ox, y + oy
                            if 0 <= xx <= 60 and 0 <= yy <= 60 and not growth[yy * 64 + xx]:
                                position = (xx, yy)
                                break
                        if position is not None:
                            break
                    if position is not None:
                        push((position[0], position[1], 2, direction,
                              flags, 0, bi, si), len(path), action)

        if "fire" in paths:
            path = paths["fire"]
            if path:
                group = fire_hit(x, y, scale, direction, flags)
                if group is not None:
                    push((x, y, scale, direction, flags | (1 << group),
                          0, bi, si), len(path), _click(path[0]))

        if "teleport" in paths:
            path = paths["teleport"]
            if path:
                if scale == 2 and big:
                    tx, ty = big[bi % len(big)]
                    nbi = (bi + 1) % len(big)
                    push((tx, ty, scale, direction, flags, 0, nbi, si),
                         len(path), _click(path[0]))
                elif scale == 1 and small:
                    tx, ty = small[si % len(small)]
                    nsi = (si + 1) % len(small)
                    push((tx, ty, scale, direction, flags, 0, bi, nsi),
                         len(path), _click(path[0]))
    return None


def act(obs, available, memory):
    obs = np.asarray(obs)
    goal, player = _find_goal_and_player(obs)
    icons = _find_icons(obs)
    mask = _panel_mask(obs)

    # Update the small-teleport cycle if our preceding click completed it.
    pending = memory.pop("pending_teleport", None)
    if pending is not None and player is not None:
        old_x, old_y, old_scale = pending
        if old_scale == 1 and (player[0], player[1]) != (old_x, old_y):
            memory["small_index"] = int(memory.get("small_index", 0)) + 1

    if goal is None or player is None:
        if "RESET" in available and not memory.get("parse_reset"):
            memory["parse_reset"] = True
            return ("RESET", None, None)
        return ("ACTION1", None, None)

    targets = _find_targets(obs, icons)
    big, small = _find_teleports(obs, icons)
    action = _search(obs, goal, player, icons, targets, big, small, mask, memory)

    if action is None:
        if "RESET" in available and not memory.get("search_reset"):
            memory["search_reset"] = True
            memory["small_index"] = 0
            return ("RESET", None, None)
        # Conservative fallback: move toward the goal.
        gx, gy, gw, gh = goal
        x, y, scale, direction = player
        if abs(gx - x) > abs(gy - y):
            action = ("ACTION4" if gx > x else "ACTION3", None, None)
        else:
            action = ("ACTION2" if gy > y else "ACTION1", None, None)

    if action[0] == "ACTION6":
        row, col = action[1], action[2]
        bit = ((row - 50) // 5) * 3 + ((col - 25) // 5)
        if 0 <= bit < 9 and (mask ^ (1 << bit)) == SPELLS["teleport"]:
            memory["pending_teleport"] = (player[0], player[1], player[2])

    if action[0] not in available:
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"):
            if name in available:
                return (name, None, None)
    return action

import heapq
import itertools
import time
from collections import Counter, deque

import numpy as np


_SIGNS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_DIRS = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}


def _geometry(frame):
    a = np.asarray(frame)
    wx = np.sum(a[1:63, 1:] != a[1:63, :-1], axis=0).astype(int)
    wy = np.sum(a[1:, :] != a[:-1, :], axis=1).astype(int)

    wx[31] = 0
    wy[0] = 0
    wy[31] = 0
    wy[62] = 0

    total = max(1, int(wx.sum() + wy.sum()))
    best = None
    for scale in range(2, 11):
        sx = [
            sum(int(wx[x - 1]) for x in range(1, 64) if x % scale == r)
            for r in range(scale)
        ]
        sy = [
            sum(int(wy[y - 1]) for y in range(1, 64) if y % scale == r)
            for r in range(scale)
        ]
        rx = max(range(scale), key=lambda r: sx[r])
        ry = max(range(scale), key=lambda r: sy[r])
        score = (sx[rx] + sy[ry]) / total + 0.004 * scale
        if best is None or score > best[0]:
            best = (score, scale, rx, ry)

    _, scale, rx, ry = best
    limit = 64 // scale

    def extent(residue):
        candidates = [
            n for n in range(1, limit + 1)
            if ((64 - n * scale) // 2) % scale == residue
        ]
        if not candidates:
            return limit, (64 - limit * scale) // 2
        n = max(candidates)
        return n, (64 - n * scale) // 2

    width, ox = extent(rx)
    height, oy = extent(ry)
    return scale, width, height, ox, oy


def _read_world(obs, memory):
    frame = np.asarray(obs)
    if "geometry" not in memory:
        memory["geometry"] = _geometry(frame)

    scale, w, h, ox, oy = memory["geometry"]
    bg = (
        (int(frame[1, 0]), int(frame[1, 63])),
        (int(frame[62, 0]), int(frame[62, 63])),
    )

    previous = memory.get("terrain")
    if previous and previous.get("size") != (w, h):
        previous = None

    old_walls = set(previous["walls"]) if previous else set()
    old_gates = dict(previous["gates"]) if previous else {}
    old_switches = dict(previous["switches"]) if previous else {}
    old_hazards = set(previous["hazards"]) if previous else set()

    walls = set()
    players = []
    player_colors = {}
    blocks = []
    selected = None
    colored = {}
    hazards = set(old_hazards)

    for y in range(h):
        for x in range(w):
            pos = y * w + x
            x0, y0 = ox + x * scale, oy + y * scale
            x1, y1 = x0 + scale, y0 + scale

            if x0 < 0 or y0 < 0 or x1 > 64 or y1 > 64:
                walls.add(pos)
                continue

            tile = frame[y0:y1, x0:x1]
            yy0 = max(y0, 1)
            yy1 = min(y1, 63)
            if yy0 >= yy1:
                walls.add(pos)
                continue

            part = frame[yy0:yy1, x0:x1]
            cy = min(62, max(1, y0 + scale // 2))
            cx = x0 + scale // 2
            center = int(frame[cy, cx])

            expected = np.empty(part.shape, dtype=np.int16)
            for iy in range(part.shape[0]):
                gy = yy0 + iy
                for ix in range(part.shape[1]):
                    gx = x0 + ix
                    expected[iy, ix] = bg[int(gy >= 32)][int(gx >= 32)]
            is_background = bool(np.all(part == expected))

            is_block = False
            if center in (9, 11) and scale >= 3:
                border = np.concatenate(
                    (tile[0, :], tile[-1, :], tile[:, 0], tile[:, -1])
                )
                is_block = np.count_nonzero(border == 5) >= len(border) * 3 // 4

            if is_block:
                blocks.append(pos)
                if center == 11:
                    selected = pos
                continue

            if not is_background and center in (1, 10):
                if np.count_nonzero(part == center) >= part.size * 3 // 4:
                    players.append(pos)
                    player_colors[pos] = center
                    continue

            if is_background:
                if pos not in old_gates:
                    walls.add(pos)
                continue

            if np.all(part == 5):
                continue

            values = np.unique(part)
            if len(values) == 1:
                colored[pos] = int(values[0])
                continue

            if np.any(part == 5):
                odd = 0
                even = 0
                for iy in range(part.shape[0]):
                    for ix in range(part.shape[1]):
                        if part[iy, ix] == 5:
                            if (x0 + ix + yy0 + iy) & 1:
                                odd += 1
                            else:
                                even += 1
                if odd and not even:
                    hazards.add(pos)
                    continue

            if pos in old_walls:
                walls.add(pos)

    gates = old_gates
    switches = old_switches
    unseen = set(colored)

    while unseen:
        start = unseen.pop()
        color = colored[start]
        component = [start]
        todo = [start]
        while todo:
            p = todo.pop()
            px, py = p % w, p // w
            for nx, ny in (
                (px - 1, py), (px + 1, py),
                (px, py - 1), (px, py + 1),
            ):
                if 0 <= nx < w and 0 <= ny < h:
                    q = ny * w + nx
                    if q in unseen and colored[q] == color:
                        unseen.remove(q)
                        component.append(q)
                        todo.append(q)

        if len(component) >= 2:
            for p in component:
                gates[p] = color
                switches.pop(p, None)
        else:
            p = component[0]
            if p not in gates:
                switches[p] = color

    walls.difference_update(gates)
    walls.difference_update(switches)

    world = {
        "size": (w, h),
        "walls": walls,
        "gates": gates,
        "switches": switches,
        "hazards": hazards,
        "blocks": tuple(sorted(blocks)),
        "selected": selected,
        "players": tuple(sorted(players)),
        "player_colors": player_colors,
        "geometry": (scale, w, h, ox, oy),
    }
    memory["terrain"] = {
        "size": (w, h),
        "walls": set(walls),
        "gates": dict(gates),
        "switches": dict(switches),
        "hazards": set(hazards),
    }
    return world


def _initial_hypotheses(world):
    visible = world["players"]
    n = len(visible)
    result = []

    if n == 2:
        for a, b in (visible, visible[::-1]):
            result.append(((a, b), 3))

    elif n == 4:
        for p in itertools.permutations(visible):
            result.append((tuple(p), 15))

    elif n == 3:
        inactive = [
            p for p in visible if world["player_colors"].get(p) == 10
        ]
        active = [
            p for p in visible if world["player_colors"].get(p) == 1
        ]
        choices = inactive if len(active) == 2 else visible
        for merged in choices:
            remaining = [p for p in visible if p != merged]
            for i, j in itertools.permutations(range(4), 2):
                p = [merged] * 4
                p[i], p[j] = remaining
                result.append((tuple(p), (1 << i) | (1 << j)))

    return result


def _visible(state):
    return tuple(sorted(set(state[0])))


def _open_colors(world, state):
    positions, mask = state
    switches = world["switches"]
    return {
        switches[p]
        for i, p in enumerate(positions)
        if mask & (1 << i) and p in switches
    }


def _advance(world, state, action, blocks=None, touches=None):
    dx, dy = _DIRS[action]
    w, h = world["size"]
    positions, mask = state
    old = list(positions)
    new = list(positions)
    blocked_blocks = set(world["blocks"] if blocks is None else blocks)
    open_colors = _open_colors(world, state)
    walls = world["walls"]
    gates = world["gates"]

    for i, p in enumerate(old):
        if not (mask & (1 << i)):
            continue
        sx, sy = _SIGNS[i]
        x = p % w + dx * sx
        y = p // w + dy * sy
        if not (0 <= x < w and 0 <= y < h):
            continue
        q = y * w + x
        if q in walls:
            continue
        if q in blocked_blocks:
            if touches is not None:
                touches[q] += 1
            continue
        if q in gates and gates[q] not in open_colors:
            continue
        new[i] = q

    for i, p in enumerate(new):
        if mask & (1 << i) and p in world["hazards"]:
            return None

    active = [i for i in range(len(old)) if mask & (1 << i)]
    for k, i in enumerate(active):
        for j in active[k + 1:]:
            if old[i] // w == old[j] // w and abs(old[i] % w - old[j] % w) == 1:
                if new[i] == old[j] or new[j] == old[i]:
                    x = ((new[i] % w) + (new[j] % w)) // 2
                    y = ((new[i] // w) + (new[j] // w)) // 2
                    new[i] = new[j] = y * w + x

    groups = {}
    for i in active:
        groups.setdefault(new[i], []).append(i)

    for members in groups.values():
        if len(members) >= 2:
            for i in members[:2]:
                mask &= ~(1 << i)
            for i in members[2:]:
                new[i] = old[i]

    return tuple(new), mask


def _heuristic(state, width):
    positions, mask = state
    active = [positions[i] for i in range(len(positions)) if mask & (1 << i)]
    if len(active) < 2:
        return 0

    def dist(a, b):
        return (
            (abs(a % width - b % width) + 1) // 2
            + (abs(a // width - b // width) + 1) // 2
        )

    if len(active) == 2:
        return dist(active[0], active[1])
    if len(active) == 4:
        a, b, c, d = active
        return min(
            dist(a, b) + dist(c, d),
            dist(a, c) + dist(b, d),
            dist(a, d) + dist(b, c),
        )
    return 0


def _search(world, start, blocks, deadline, limit=50000):
    width = world["size"][0]
    touches = Counter()
    queue = []
    sequence = itertools.count()
    heapq.heappush(queue, (0, 0, next(sequence), start))
    best = {start: 0}
    parent = {}
    expanded = 0

    while queue and expanded < limit:
        if expanded % 256 == 0 and time.monotonic() >= deadline:
            break

        _, cost, _, state = heapq.heappop(queue)
        if best.get(state) != cost:
            continue

        if state[1] == 0:
            path = []
            while state != start:
                state, action = parent[state]
                path.append(action)
            path.reverse()
            return path, touches

        expanded += 1
        for action in _DIRS:
            nxt = _advance(world, state, action, blocks, touches)
            if nxt is None or nxt == state:
                continue
            nc = cost + 1
            if nc >= best.get(nxt, 10**9):
                continue
            best[nxt] = nc
            parent[nxt] = (state, action)
            priority = nc + 1.35 * _heuristic(nxt, width)
            heapq.heappush(queue, (priority, nc, next(sequence), nxt))

    return None, touches


def _click(world, p):
    scale, w, _, ox, oy = world["geometry"]
    return (
        "ACTION6",
        oy + (p // w) * scale + scale // 2,
        ox + (p % w) * scale + scale // 2,
    )


def _route_cells(world, state, path):
    """Cells occupied by pieces on a route with blockers removed."""
    cells = set(state[0])
    for action in path:
        state = _advance(world, state, action, ())
        if state is None:
            break
        cells.update(state[0])
    return cells


def _parking_route(world, state, block, reserved):
    """Find a route to park a blocker outside the intended piece route."""
    w, h = world["size"]
    open_colors = _open_colors(world, state)
    forbidden = set(world["walls"])
    forbidden.update(world["switches"])
    forbidden.update(world["hazards"])
    forbidden.update(state[0])
    forbidden.update(p for p in world["blocks"] if p != block)
    forbidden.update(
        p for p, color in world["gates"].items()
        if color not in open_colors
    )

    queue = deque([block])
    parent = {block: None}
    destination = None

    while queue:
        p = queue.popleft()
        if p != block and p not in reserved:
            destination = p
            break
        x, y = p % w, p // w
        for action, (dx, dy) in _DIRS.items():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            q = ny * w + nx
            if q in forbidden or q in parent:
                continue
            parent[q] = (p, action)
            queue.append(q)

    if destination is None:
        return None

    path = []
    p = destination
    while p != block:
        p, action = parent[p]
        path.append(action)
    path.reverse()
    return destination, path


def _relaxed_relocation(world, state, deadline):
    """Find one blocker relocation supporting a blocker-free solution."""
    path, _ = _search(world, state, (), deadline, limit=70000)
    if not path:
        return None

    reserved = _route_cells(world, state, path)
    blocking = [p for p in world["blocks"] if p in reserved]
    if not blocking:
        return None

    choices = []
    for block in blocking:
        parking = _parking_route(world, state, block, reserved)
        if parking is not None:
            destination, route = parking
            choices.append((len(route), block, destination, route))

    if not choices:
        return None

    _, block, destination, route = min(choices)
    return block, destination, route


def _relocations(world, state, touches):
    """Fallback short blocker relocations."""
    w, h = world["size"]
    blocks = world["blocks"]
    open_colors = _open_colors(world, state)
    forbidden = set(world["walls"]) | set(world["switches"])
    forbidden |= set(world["hazards"]) | set(state[0])
    forbidden |= {
        p for p, c in world["gates"].items() if c not in open_colors
    }

    candidates = []
    for block in sorted(blocks, key=lambda p: -touches[p]):
        occupied = forbidden | (set(blocks) - {block})
        queue = deque([(block, ())])
        seen = {block}

        while queue:
            p, path = queue.popleft()
            if path:
                candidates.append((block, p, path, touches[block]))
            if len(path) >= 4:
                continue
            x, y = p % w, p // w
            for action, (dx, dy) in _DIRS.items():
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                q = ny * w + nx
                if q in seen or q in occupied:
                    continue
                seen.add(q)
                queue.append((q, path + (action,)))

    candidates.sort(key=lambda item: (-item[3], len(item[2])))
    return candidates


def _emit(world, memory, action, chosen=None):
    result = (action, None, None) if isinstance(action, str) else action
    name = result[0]
    hypotheses = memory.get("hypotheses", [])

    if name in _DIRS and world["selected"] is None:
        predicted = []
        for state in hypotheses:
            nxt = _advance(world, state, name)
            if nxt is not None:
                predicted.append(nxt)
        memory["hypotheses"] = list(dict.fromkeys(predicted))
        if chosen is not None:
            chosen = _advance(world, chosen, name)

    memory["chosen"] = chosen
    memory["expected"] = _visible(chosen) if chosen is not None else None
    return result


def _relocation_actions(world, state, block, route):
    positions, mask = state
    target = next(
        (positions[i] for i in range(len(positions)) if mask & (1 << i)),
        positions[0],
    )
    actions = [_click(world, block)]
    actions.extend(route)
    actions.append(_click(world, target))
    return actions


def act(obs, available, memory):
    deadline = time.monotonic() + 7.5
    world = _read_world(obs, memory)
    visible = world["players"]

    hypotheses = memory.get("hypotheses")
    if hypotheses:
        hypotheses = [s for s in hypotheses if _visible(s) == visible]
    if not hypotheses:
        hypotheses = _initial_hypotheses(world)
        memory["plan"] = []
        memory["chosen"] = None
    memory["hypotheses"] = hypotheses

    expected = memory.get("expected")
    if expected is not None and expected != visible:
        memory["plan"] = []
        memory["chosen"] = None

    chosen = memory.get("chosen")
    if chosen not in hypotheses:
        chosen = None
        memory["plan"] = []

    plan = memory.get("plan", [])
    if plan and chosen is not None:
        action = plan.pop(0)
        memory["plan"] = plan
        return _emit(world, memory, action, chosen)

    if world["selected"] is not None:
        target = visible[0] if visible else 0
        return _emit(
            world, memory, _click(world, target),
            hypotheses[0] if hypotheses else None,
        )

    if not hypotheses:
        return ("ACTION1", None, None)

    touch_records = []

    # Try solving without moving blockers.
    for state in hypotheses[:min(4, len(hypotheses))]:
        local_deadline = min(deadline, time.monotonic() + 1.0)
        path, touches = _search(
            world, state, world["blocks"], local_deadline, limit=45000
        )
        touch_records.append((state, touches))
        if path:
            memory["plan"] = path[1:]
            return _emit(world, memory, path[0], state)
        if time.monotonic() >= deadline - 4.0:
            break

    # Find a route without blockers, then park a blocker outside that route.
    # Replanning after each relocation supports moving several blockers.
    if world["blocks"]:
        for state in hypotheses[:min(4, len(hypotheses))]:
            local_deadline = min(deadline, time.monotonic() + 1.2)
            relocation = _relaxed_relocation(world, state, local_deadline)
            if relocation is not None:
                block, _, route = relocation
                actions = _relocation_actions(world, state, block, route)
                memory["plan"] = actions[1:]
                return _emit(world, memory, actions[0], state)
            if time.monotonic() >= deadline - 1.0:
                break

    # Fallback: test short relocations directly.
    for state, touches in touch_records[:2]:
        for block, destination, route, _ in _relocations(world, state, touches):
            if time.monotonic() >= deadline - 0.1:
                break
            new_blocks = tuple(sorted(
                destination if p == block else p for p in world["blocks"]
            ))
            local_deadline = min(deadline, time.monotonic() + 0.35)
            path, _ = _search(
                world, state, new_blocks, local_deadline, limit=25000
            )
            if path is None:
                continue
            actions = _relocation_actions(world, state, block, route)
            actions.extend(path)
            memory["plan"] = actions[1:]
            return _emit(world, memory, actions[0], state)

    # Probe a direction to distinguish remaining control assignments.
    best_action = None
    best_score = -1
    for action in _DIRS:
        outcomes = []
        moves = 0
        for state in hypotheses:
            nxt = _advance(world, state, action)
            if nxt is not None:
                outcomes.append(_visible(nxt))
                if nxt != state:
                    moves += 1
        score = len(set(outcomes)) * 100 + moves
        if score > best_score:
            best_score = score
            best_action = action

    if best_action is None:
        best_action = "ACTION1"
    memory["plan"] = []
    return _emit(world, memory, best_action, None)

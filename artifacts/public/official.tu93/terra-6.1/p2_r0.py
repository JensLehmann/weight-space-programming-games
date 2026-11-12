from collections import deque
import heapq
import itertools
import numpy as np


# Direction order: up, down, left, right.
_NAMES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_STRIDE = 256


def _pack(x, y):
    return (y + 64) * _STRIDE + x + 64


def _unpack(p):
    return p % _STRIDE - 64, p // _STRIDE - 64


def _components(mask):
    """Four-connected components of a small Boolean image."""
    points = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while points:
        start = points.pop()
        stack = [start]
        component = [start]
        while stack:
            y, x = stack.pop()
            for q in ((y - 1, x), (y + 1, x),
                      (y, x - 1), (y, x + 1)):
                if q in points:
                    points.remove(q)
                    stack.append(q)
                    component.append(q)
        result.append(component)
    return result


def _find_player(frame):
    """Return scale, center, and facing of the player."""
    candidates = []
    for comp in _components(frame == 4):
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        my, mx = min(ys), min(xs)
        h, w = max(ys) - my + 1, max(xs) - mx + 1
        if h != w or len(comp) != h * w or h < 1:
            continue
        s = h

        # Marker tile is respectively at top, bottom, left, or right.
        origins = (
            (mx - s, my),
            (mx - s, my - 2 * s),
            (mx, my - s),
            (mx - 2 * s, my - s),
        )
        for direction, (x, y) in enumerate(origins):
            if x < 0 or y < 0 or x + 3 * s > 64 or y + 3 * s > 63:
                continue
            good = True
            for ty in range(3):
                for tx in range(3):
                    marker = ((direction == 0 and (tx, ty) == (1, 0)) or
                              (direction == 1 and (tx, ty) == (1, 2)) or
                              (direction == 2 and (tx, ty) == (0, 1)) or
                              (direction == 3 and (tx, ty) == (2, 1)))
                    color = 4 if marker else 9
                    tile = frame[y + ty * s:y + (ty + 1) * s,
                                 x + tx * s:x + (tx + 1) * s]
                    if not np.all(tile == color):
                        good = False
                        break
                if not good:
                    break
            if good:
                cx = x + s + s // 2
                cy = y + s + s // 2
                candidates.append((s, cx, cy, direction))
    if not candidates:
        return None
    return max(candidates)


def _read_frame(frame, player):
    s, px, py, player_direction = player
    spacing = 6 * s
    xs = list(range(px % spacing, 64, spacing))
    ys = list(range(py % spacing, 63, spacing))
    deltas = tuple((dy * _STRIDE + dx) * spacing for dx, dy in _DIRS)

    graph = {}
    greens, reds, purples = [], [], []
    goals = set()
    visible = {}

    for y in ys:
        for x in xs:
            p = _pack(x, y)
            edges = {}
            for d, (dx, dy) in enumerate(_DIRS):
                nx, ny = x + dx * spacing, y + dy * spacing
                mx, my = x + dx * 3 * s, y + dy * 3 * s
                if (0 <= nx < 64 and 0 <= ny < 63 and
                        0 <= mx < 64 and 0 <= my < 63 and
                        frame[my, mx] == 2):
                    edges[d] = p + deltas[d]
            if edges:
                graph[p] = edges

            color = int(frame[y, x])
            if color == 14:
                goals.add(p)
                continue
            if color not in (8, 12, 13):
                continue

            direction = None
            marker = None
            for d, (dx, dy) in enumerate(_DIRS):
                ax, ay = x + dx * s, y + dy * s
                if 0 <= ax < 64 and 0 <= ay < 63:
                    c = int(frame[ay, ax])
                    if c in (11, 15):
                        direction, marker = d, c
                        break
            if direction is None:
                continue

            if color == 8:
                greens.append((p, direction))
                visible[p] = (8, direction, None)
            elif color == 12:
                reds.append((p, direction))
                visible[p] = (12, direction, None)
            else:
                # For an already activated follower, its unseen pending
                # direction is usually the player's previous direction.
                pending = player_direction if marker == 11 else -1
                purples.append((p, direction, pending))
                visible[p] = (13, direction, marker == 11)

    state = (_pack(px, py), tuple(sorted(greens)),
             tuple(sorted(reds)), tuple(sorted(purples)))
    return graph, deltas, goals, visible, state


def _matches_prediction(pred, observed, visible):
    """Allow predicted enemies to be hidden behind one another."""
    if not pred or pred[0] != observed[0]:
        return False
    by_position = {}
    for p, d in pred[1]:
        by_position.setdefault(p, []).append((8, d, None))
    for p, d in pred[2]:
        by_position.setdefault(p, []).append((12, d, None))
    for p, d, pending in pred[3]:
        by_position.setdefault(p, []).append((13, d, pending != -1))

    # An enemy whose center is off-screen cannot be checked visually.
    on_screen = {}
    for p, entries in by_position.items():
        x, y = _unpack(p)
        if 0 <= x < 64 and 0 <= y < 63:
            on_screen[p] = entries

    if set(on_screen) != set(visible):
        return False
    return all(item in on_screen[p] for p, item in visible.items())


def _advance(state, direction, graph, deltas):
    """Simulate one valid action. Return None if the player dies."""
    player, greens, reds, purples = state
    next_player = graph.get(player, {}).get(direction)
    if next_player is None:
        return None

    # The player removes any enemies at the destination before they move.
    greens = tuple(e for e in greens if e[0] != next_player)
    reds = tuple(e for e in reds if e[0] != next_player)
    purples = tuple(e for e in purples if e[0] != next_player)

    # A green enemy fires when the player is one node directly in front.
    for p, d in greens:
        if p + deltas[d] == next_player:
            return None

    new_reds = []
    for p, d in reds:
        q = p + deltas[d]
        if q == next_player:
            return None
        # After moving, patrols reverse if there is no track ahead.
        if d not in graph.get(q, {}):
            d = (1, 0, 3, 2)[d]
        new_reds.append((q, d))

    new_purples = []
    for p, d, pending in purples:
        if pending != -1:
            q = p + deltas[d]
            if q == next_player:
                return None
            new_purples.append((q, pending, direction))
        else:
            # Followers activate when the player is two nodes ahead.
            if p + 2 * deltas[d] == next_player:
                new_purples.append((p, d, d))
            else:
                new_purples.append((p, d, -1))

    return (next_player, tuple(sorted(greens)),
            tuple(sorted(new_reds)), tuple(sorted(new_purples)))


def _distances(graph, goals):
    distances = {}
    queue = deque()
    for p in goals:
        if p in graph:
            distances[p] = 0
            queue.append(p)
    while queue:
        p = queue.popleft()
        nd = distances[p] + 1
        for q in graph.get(p, {}).values():
            if q not in distances:
                distances[q] = nd
                queue.append(q)
    return distances


def _path_to(state, parent):
    path = []
    while state in parent:
        previous, direction = parent[state]
        path.append(direction)
        state = previous
    path.reverse()
    return path


def _search(initial, graph, deltas, goals, allowed):
    """A* with track distance as an admissible heuristic."""
    distances = _distances(graph, goals)
    if initial[0] not in distances:
        return None

    counter = itertools.count()
    heap = [(distances[initial[0]], 0, next(counter), initial)]
    best_cost = {initial: 0}
    parent = {}
    best_progress = initial
    best_progress_key = (distances[initial[0]], 0)
    expanded = 0
    limit = 90000

    while heap and expanded < limit:
        _, cost, _, state = heapq.heappop(heap)
        if best_cost.get(state) != cost:
            continue
        if cost and state[0] in goals:
            return _path_to(state, parent)

        expanded += 1
        for direction in allowed:
            if direction not in graph.get(state[0], {}):
                continue
            nxt = _advance(state, direction, graph, deltas)
            if nxt is None or nxt[0] not in distances:
                continue
            new_cost = cost + 1
            if new_cost >= best_cost.get(nxt, 10**9):
                continue
            best_cost[nxt] = new_cost
            parent[nxt] = (state, direction)
            h = distances[nxt[0]]
            key = (h, new_cost)
            if key < best_progress_key:
                best_progress_key = key
                best_progress = nxt
            heapq.heappush(
                heap, (new_cost + h, new_cost, next(counter), nxt)
            )

    # If the search limit was reached, make safe progress and replan.
    if best_progress != initial:
        return _path_to(best_progress, parent)
    return None


def act(obs, available, memory):
    frame = np.asarray(obs)
    allowed = [i for i, name in enumerate(_NAMES) if name in available]
    if not allowed:
        if "RESET" in available:
            return ("RESET", None, None)
        return (available[0], None, None)

    player = _find_player(frame)
    if player is None:
        # This should only occur for a terminal or unrecognizable frame.
        if "RESET" in available:
            memory.clear()
            return ("RESET", None, None)
        return (_NAMES[allowed[0]], None, None)

    graph, deltas, seen_goals, visible, observed = _read_frame(frame, player)

    # Goals can temporarily be covered by an enemy.
    goals = set(memory.get("goals", []))
    goals.update(seen_goals)
    memory["goals"] = list(goals)

    predicted = memory.get("expected")
    prediction_matches = _matches_prediction(predicted, observed, visible)
    state = predicted if prediction_matches else observed

    visits = memory.setdefault("visits", {})
    pos_key = str(state[0])
    visits[pos_key] = visits.get(pos_key, 0) + 1

    path = None
    if prediction_matches:
        cached = memory.get("plan", [])
        if cached and cached[0] in allowed:
            if _advance(state, cached[0], graph, deltas) is not None:
                path = list(cached)

    if path is None and goals:
        path = _search(state, graph, deltas, goals, allowed)

    if not path:
        # If the goal is hidden or search failed, choose a safe move,
        # preferring less-visited positions and visible progress.
        distances = _distances(graph, goals)
        options = []
        for direction in allowed:
            nxt = _advance(state, direction, graph, deltas)
            if nxt is None:
                continue
            p = nxt[0]
            options.append((
                distances.get(p, 1000),
                visits.get(str(p), 0),
                direction,
                nxt,
            ))
        if not options:
            if "RESET" in available:
                memory.clear()
                return ("RESET", None, None)
            return (_NAMES[allowed[0]], None, None)
        options.sort(key=lambda item: item[:3])
        direction = options[0][2]
        memory["expected"] = options[0][3]
        memory["plan"] = []
        return (_NAMES[direction], None, None)

    direction = path[0]
    memory["expected"] = _advance(state, direction, graph, deltas)
    memory["plan"] = path[1:]
    return (_NAMES[direction], None, None)

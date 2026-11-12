import heapq
import time
from collections import deque

import numpy as np


# Directions correspond to ACTION1 through ACTION4.
_STRIDE = 256
_PAD = 100
_DELTA = (0, -_STRIDE, _STRIDE, -1, 1)
_REVERSE = (0, 2, 1, 4, 3)
_EMPTY = (None, None, None, None)
_MARKERS = ((0, 1, 1), (2, 1, 2), (1, 0, 3), (1, 2, 4))


def _pack(position, direction, pending=0):
    return (position << 6) | (pending << 3) | direction


def _position_distance(a, b):
    return (
        abs(a // _STRIDE - b // _STRIDE)
        + abs(a % _STRIDE - b % _STRIDE)
    )


def _read_board(obs):
    yy, xx = np.where(obs[:63] == 9)
    if len(xx) == 0:
        return None

    px, py = int(xx.min()), int(yy.min())
    width = int(xx.max()) - px + 1
    height = int(yy.max()) - py + 1
    if width != height or width % 3 or width < 3:
        return None

    scale = width // 3
    size = 3 * scale
    spacing = 6 * scale
    bx, by = px % spacing, py % spacing

    def encode(x, y):
        gx = (x - bx) // spacing + _PAD
        gy = (y - by) // spacing + _PAD
        return gy * _STRIDE + gx

    player = encode(px, py)
    graph = {}
    goals = set()
    actors = []

    screen_delta = (
        (0, -size), (0, size), (-size, 0), (size, 0)
    )

    for y in range(by, 64 - size + 1, spacing):
        for x in range(bx, 64 - size + 1, spacing):
            p = encode(x, y)
            patch = obs[y:y + size, x:x + size]

            # Rails lie halfway between consecutive nodes.
            for index, (dx, dy) in enumerate(screen_delta):
                ex, ey = x + dx, y + dy
                if (
                    0 <= ex <= 64 - size
                    and 0 <= ey <= 63 - size
                    and np.all(obs[ey:ey + size, ex:ex + size] == 2)
                ):
                    action = index + 1
                    q = p + _DELTA[action]
                    graph.setdefault(p, [None] * 4)[index] = q
                    graph.setdefault(q, [None] * 4)[
                        _REVERSE[action] - 1
                    ] = p

            if np.all(patch == 14):
                goals.add(p)
                graph.setdefault(p, [None] * 4)
                continue

            base = int(patch[0, 0])
            if base not in (8, 12, 13):
                continue

            small = patch[::scale, ::scale]
            if small.shape != (3, 3):
                continue

            for mr, mc, direction in _MARKERS:
                marker = int(small[mr, mc])
                if marker not in (11, 15):
                    continue
                expected = np.full((3, 3), base, dtype=np.int16)
                expected[mr, mc] = marker
                if np.array_equal(small, expected):
                    # Also verify the interiors of enlarged pixel blocks.
                    expanded = np.repeat(
                        np.repeat(expected, scale, axis=0),
                        scale, axis=1
                    )
                    if np.array_equal(patch, expanded):
                        actors.append((base, p, direction, marker == 11))
                        graph.setdefault(p, [None] * 4)
                    break

    graph.setdefault(player, [None] * 4)
    graph = {p: tuple(adj) for p, adj in graph.items()}
    signature = tuple(
        sorted(
            (p, adj)
            for p, adj in graph.items()
            if any(q is not None for q in adj)
        )
    )
    actors.sort()
    origin = (bx, by, scale)
    return graph, goals, player, actors, origin, signature


def _initial_model(player, actors):
    kinds = []
    codes = []
    for kind, p, direction, active in actors:
        kinds.append(kind)
        pending = direction if kind == 13 and active else 0
        codes.append(_pack(p, direction, pending))
    return tuple(kinds), (player,) + tuple(codes)


def _reconcile(player, actors, kinds, predicted, origin):
    """Associate visible enemies with the previous prediction."""
    kinds = list(kinds)
    codes = list(predicted[1:])
    unused = {i for i, code in enumerate(codes) if code != -1}
    observed_positions = {actor[1] for actor in actors}

    for kind, p, direction, active in actors:
        candidates = [i for i in unused if kinds[i] == kind]
        if candidates:
            i = min(
                candidates,
                key=lambda j: (
                    _position_distance(codes[j] >> 6, p),
                    (codes[j] & 7) != direction,
                    j
                )
            )
            unused.remove(i)
            old_pending = (codes[i] >> 3) & 7
            if kind == 13 and active:
                pending = old_pending or direction
            else:
                pending = 0
            codes[i] = _pack(p, direction, pending)
        else:
            kinds.append(kind)
            pending = direction if kind == 13 and active else 0
            codes.append(_pack(p, direction, pending))

    # Retain enemies hidden underneath another enemy or outside the viewport.
    bx, by, scale = origin
    spacing, size = 6 * scale, 3 * scale
    for i in unused:
        p = codes[i] >> 6
        if p in observed_positions:
            continue
        x = (p % _STRIDE - _PAD) * spacing + bx
        y = (p // _STRIDE - _PAD) * spacing + by
        if 0 <= x <= 64 - size and 0 <= y <= 63 - size:
            codes[i] = -1

    return tuple(kinds), (player,) + tuple(codes)


def _advance(state, action, kinds, graph):
    target = graph.get(state[0], _EMPTY)[action - 1]
    if target is None:
        return None

    result = [target]
    for kind, code in zip(kinds, state[1:]):
        if code == -1:
            result.append(-1)
            continue

        p = code >> 6
        direction = code & 7
        pending = (code >> 3) & 7

        # The player completes its movement before enemies move.
        # Entering an enemy's current node destroys that enemy.
        if p == target:
            result.append(-1)
            continue

        if kind == 8:
            # Stationary enemy attacks the immediately forward node.
            if p + _DELTA[direction] == target:
                return None
            result.append(code)

        elif kind == 12:
            # Patrol moves one node and then reverses at a rail endpoint.
            np_ = p + _DELTA[direction]
            if np_ == target:
                return None
            if graph.get(np_, _EMPTY)[direction - 1] is None:
                direction = _REVERSE[direction]
            result.append(_pack(np_, direction))

        else:  # kind == 13
            if pending:
                np_ = p + _DELTA[direction]
                if np_ == target:
                    return None
                # Current direction, then queued direction, then this action.
                result.append(_pack(np_, pending, action))
            else:
                # Activates when the player is two nodes directly ahead.
                if p + 2 * _DELTA[direction] == target:
                    pending = direction
                result.append(_pack(p, direction, pending))

    return tuple(result)


def _distances(graph, goals):
    distance = {p: 0 for p in goals}
    queue = deque(goals)
    while queue:
        p = queue.popleft()
        nd = distance[p] + 1
        for q in graph.get(p, _EMPTY):
            if q is not None and q not in distance:
                distance[q] = nd
                queue.append(q)
    return distance


def _remaining_steps(obs, mem):
    filled = int(np.count_nonzero(obs[63] == 6))
    previous = mem.get("timers")
    candidates = []

    if previous:
        for total, remaining in previous:
            remaining -= 1
            if remaining >= 0 and round(64 * remaining / total) == filled:
                candidates.append((total, remaining))

    if not candidates:
        # Infer possible timer denominators rather than assuming a level.
        for total in range(1, 257):
            approximate = filled * total // 64
            for remaining in range(
                max(0, approximate - 2),
                min(total, approximate + 2) + 1
            ):
                if round(64 * remaining / total) == filled:
                    candidates.append((total, remaining))

    mem["timers"] = candidates
    return max((r for _, r in candidates), default=100)


def _extract_plan(end, parents):
    plan = []
    while end in parents:
        end, action = parents[end]
        plan.append(action)
    plan.reverse()
    return plan


def _search(start, kinds, graph, goals, allowed, distance,
            remaining, deadline):
    if start[0] not in distance:
        return None, True
    if start[0] in goals:
        return [], False

    h0 = distance[start[0]]
    if h0 > remaining:
        return None, True

    weight = 1.4
    serial = 0
    heap = [(weight * h0, h0, 0, serial, start)]
    costs = {start: 0}
    parents = {}
    best = start
    best_rank = (h0, 0)
    popped = 0
    interrupted = False

    while heap:
        if popped % 128 == 0:
            if time.perf_counter() >= deadline or popped >= 180000:
                interrupted = True
                break

        _, h, g, _, state = heapq.heappop(heap)
        popped += 1
        if costs.get(state) != g:
            continue

        if state[0] in goals:
            return _extract_plan(state, parents), False

        if g >= remaining:
            continue

        successors = []
        for action in allowed:
            ns = _advance(state, action, kinds, graph)
            if ns is None:
                continue
            nh = distance.get(ns[0])
            if nh is None:
                continue
            ng = g + 1
            if ng + nh > remaining:
                continue
            successors.append((action, ns, ng, nh))

        # A partial route should end somewhere with a safe continuation.
        if successors and (h, g) < best_rank:
            best, best_rank = state, (h, g)

        for action, ns, ng, nh in successors:
            if ng >= costs.get(ns, 10**9):
                continue
            costs[ns] = ng
            parents[ns] = (state, action)
            serial += 1
            heapq.heappush(
                heap, (ng + weight * nh, nh, ng, serial, ns)
            )

    if interrupted and best != start:
        return _extract_plan(best, parents), False
    return None, not interrupted


def act(obs, available, memory):
    started = time.perf_counter()
    names = set(available)
    allowed = tuple(
        a for a in range(1, 5) if "ACTION%d" % a in names
    )
    mem = memory.setdefault("_tu93", {})

    def reset_or_fallback():
        if "RESET" in names:
            mem.clear()
            return "RESET", None, None
        if allowed:
            return "ACTION%d" % allowed[0], None, None
        return (available[0] if available else "ACTION1"), None, None

    board = _read_board(obs)
    if board is None or not allowed:
        return reset_or_fallback()

    graph, visible_goals, player, actors, origin, signature = board
    same_board = (
        mem.get("origin") == origin
        and mem.get("edges") == signature
    )

    if not same_board:
        mem.clear()

    remaining = _remaining_steps(obs, mem)
    goals = set(visible_goals)
    if same_board:
        goals.update(mem.get("goals", ()))

    predicted = mem.get("predicted")
    if (
        same_board
        and predicted is not None
        and predicted[0] == player
        and "kinds" in mem
    ):
        kinds, state = _reconcile(
            player, actors, mem["kinds"], predicted, origin
        )
    else:
        kinds, state = _initial_model(player, actors)

    cached = list(mem.get("plan", ()))
    if state != predicted or kinds != mem.get("kinds"):
        cached = []

    mem["origin"] = origin
    mem["edges"] = signature
    mem["goals"] = tuple(sorted(goals))

    visits = mem.setdefault("visits", {})
    visits[state] = visits.get(state, 0) + 1
    if len(visits) > 1000:
        visits.clear()
        visits[state] = 1

    if cached and len(cached) <= remaining:
        action = cached[0]
        if (
            action in allowed
            and _advance(state, action, kinds, graph) is not None
        ):
            mem["kinds"] = kinds
            mem["predicted"] = _advance(state, action, kinds, graph)
            mem["plan"] = cached[1:]
            return "ACTION%d" % action, None, None

    search_goals = goals
    if not search_goals:
        # An enemy can temporarily cover the exit. Prefer covered endpoints
        # while moving until the exit becomes visible.
        leaves = {
            p for p, adj in graph.items()
            if sum(q is not None for q in adj) == 1 and p != player
        }
        occupied = {actor[1] for actor in actors}
        search_goals = leaves & occupied
        if not search_goals:
            search_goals = leaves
        if not search_goals:
            search_goals = set(graph) - {player}

    distance = _distances(graph, search_goals)
    plan, exhausted = _search(
        state, kinds, graph, search_goals, allowed, distance,
        max(0, remaining), started + 7.3
    )

    if plan:
        action = plan[0]
        suffix = plan[1:]
    else:
        if exhausted and goals and "RESET" in names:
            return reset_or_fallback()

        # Safe local fallback, with a small anti-cycle preference.
        choices = []
        for action in allowed:
            ns = _advance(state, action, kinds, graph)
            if ns is None:
                continue
            continuations = sum(
                _advance(ns, b, kinds, graph) is not None
                for b in allowed
            )
            score = (
                distance.get(ns[0], 1000)
                + 3 * visits.get(ns, 0)
                + (4 if continuations == 0 else 0),
                -continuations,
                action
            )
            choices.append((score, action, ns))

        if not choices:
            return reset_or_fallback()
        _, action, _ = min(choices)
        suffix = []

    mem["kinds"] = kinds
    mem["predicted"] = _advance(state, action, kinds, graph)
    mem["plan"] = suffix
    return "ACTION%d" % action, None, None

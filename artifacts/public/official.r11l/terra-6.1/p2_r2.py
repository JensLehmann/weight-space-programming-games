from collections import Counter, deque
from functools import lru_cache

import numpy as np


_CIRCLE = tuple(
    (x, y)
    for y in range(-2, 3)
    for x in range(-2, 3)
    if not (abs(x) == 2 and abs(y) == 2)
)

_CROSS = tuple(
    (x, y)
    for y in range(-2, 3)
    for x in range(-2, 3)
    if abs(x) + abs(y) <= 2
)

_RING = (
    (-1, -3), (1, -3),
    (-2, -2), (2, -2),
    (-3, -1), (3, -1),
    (-3, 1), (3, 1),
    (-2, 2), (2, 2),
    (-1, 3), (1, 3),
)

_GOAL = tuple(
    (x, y)
    for y in range(-3, 4)
    for x in range(-3, 4)
    if (
        (abs(y) == 3 and abs(x) <= 1)
        or (abs(y) == 2 and abs(x) <= 2)
        or (abs(y) <= 1 and abs(x) <= 3)
    )
)

_PIECE_LEFT = {
    (-1, -2),
    (-2, -1), (-1, -1),
    (-2, 0), (-1, 0),
    (-2, 1), (-1, 1), (0, 1),
    (-1, 2), (0, 2),
}
_PIECE_RIGHT = set(_CIRCLE) - _PIECE_LEFT

_PIECE_HITS = {
    (ax - bx, ay - by)
    for ax, ay in _CIRCLE
    for bx, by in _CIRCLE
}
_GOAL_HITS = {
    (ax - bx, ay - by)
    for ax, ay in _GOAL
    for bx, by in _CIRCLE
}


def _line_points(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    points = []

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return points
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _line_score(obs, marker, center):
    points = _line_points(
        marker["x"], marker["y"], center[0], center[1]
    )[3:-3]
    if not points:
        return 0.0
    return sum(int(obs[y, x]) == 1 for x, y in points) / len(points)


def _find_markers(obs):
    markers = []
    forbidden = {0, 1, 2, 3, 5, 6, 10}

    for y in range(2, 62):
        for x in range(2, 62):
            color = int(obs[y, x])
            if color in forbidden:
                continue

            neighbors = [
                int(obs[y + dy, x + dx])
                for dx, dy in _CROSS
                if dx or dy
            ]
            n0 = neighbors.count(0)
            n3 = neighbors.count(3)

            if max(n0, n3) >= 9:
                markers.append({
                    "x": x,
                    "y": y,
                    "color": color,
                    "selected": n0 > n3,
                })

    return markers


def _find_goals(obs, marker_colors):
    raw = []
    forbidden = {0, 1, 2, 3, 4, 5, 6, 10}
    left = _RING[0::2]
    right = _RING[1::2]

    for y in range(3, 61):
        for x in range(3, 61):
            a = Counter(
                int(obs[y + dy, x + dx])
                for dx, dy in left
                if int(obs[y + dy, x + dx]) not in forbidden
            )
            b = Counter(
                int(obs[y + dy, x + dx])
                for dx, dy in right
                if int(obs[y + dy, x + dx]) not in forbidden
            )
            if not a or not b:
                continue

            ca, na = a.most_common(1)[0]
            cb, nb = b.most_common(1)[0]
            if na < 4 or nb < 4:
                continue

            colors = frozenset((ca, cb))
            if len(colors) == 1 and ca not in marker_colors:
                continue

            raw.append((na + nb, x, y, colors))

    raw.sort(reverse=True)
    goals = []

    for score, x, y, colors in raw:
        if any(
            abs(x - goal["x"]) <= 2 and abs(y - goal["y"]) <= 2
            for goal in goals
        ):
            continue
        goals.append({"x": x, "y": y, "colors": set(colors)})

    return goals


def _ball_at(obs, x, y, marker_color):
    if not (2 <= x <= 61 and 2 <= y <= 61):
        return None

    values = [int(obs[y + dy, x + dx]) for dx, dy in _CIRCLE]

    if marker_color != 4:
        if int(obs[y, x]) != 6:
            return None
        if values.count(marker_color) < 15:
            return None
        return set()

    bad = sum(v in (1, 2, 3, 4, 5, 6, 10) for v in values)
    if bad > 2:
        return None

    return {
        v for v in values
        if v not in (0, 1, 2, 3, 4, 5, 6, 10)
    }


def _make_groups(obs, markers):
    groups = []

    for color in sorted({m["color"] for m in markers}):
        ids = [i for i, m in enumerate(markers) if m["color"] == color]
        if len(ids) > 12:
            ids = ids[:12]
        n = len(ids)
        if not n:
            continue

        options = [[] for _ in range(n)]

        for mask in range(1, 1 << n):
            members = [ids[j] for j in range(n) if mask & (1 << j)]
            cx = sum(markers[i]["x"] for i in members) // len(members)
            cy = sum(markers[i]["y"] for i in members) // len(members)

            colors = _ball_at(obs, cx, cy, color)
            if colors is None:
                continue

            score = 20.0 * len(members)
            score += sum(
                _line_score(obs, markers[i], (cx, cy))
                for i in members
            )
            score += 2.0 if len(members) >= 2 else -2.0
            first = (mask & -mask).bit_length() - 1

            options[first].append(
                (mask, members, tuple(sorted(colors)), score)
            )

        @lru_cache(None)
        def solve(remaining):
            if not remaining:
                return 0.0, ()

            first = (remaining & -remaining).bit_length() - 1
            rest = remaining & ~(1 << first)
            best_score, best_groups = solve(rest)
            best_score -= 5.0

            for mask, members, colors, score in options[first]:
                if mask & remaining != mask:
                    continue
                sub_score, sub_groups = solve(remaining ^ mask)
                total = score + sub_score
                if total > best_score:
                    best_score = total
                    best_groups = (
                        (tuple(members), colors),
                    ) + sub_groups

            return best_score, best_groups

        _, found = solve((1 << n) - 1)
        for members, colors in found:
            groups.append({
                "markers": list(members),
                "kind": color,
                "colors": set(colors),
            })

    return groups


def _center(group, markers):
    ids = group["markers"]
    return (
        sum(markers[i]["x"] for i in ids) // len(ids),
        sum(markers[i]["y"] for i in ids) // len(ids),
    )


def _find_pieces(obs, groups, markers):
    ball_centers = {_center(group, markers) for group in groups}
    pieces = []

    for y in range(2, 62):
        for x in range(2, 62):
            if (x, y) in ball_centers:
                continue

            vals = {
                (dx, dy): int(obs[y + dy, x + dx])
                for dx, dy in _CIRCLE
            }
            nonzero = {v for v in vals.values() if v != 0}
            if len(nonzero) != 1:
                continue

            color = next(iter(nonzero))
            if color in (1, 2, 3, 4, 5, 6, 10):
                continue

            colored = {p for p, v in vals.items() if v == color}
            if colored == _PIECE_LEFT or colored == _PIECE_RIGHT:
                pieces.append({"x": x, "y": y, "color": color})

    return pieces


def _hit(x, y, obj, goal=False):
    offsets = _GOAL_HITS if goal else _PIECE_HITS
    return (x - obj["x"], y - obj["y"]) in offsets


def _initialize(obs, memory):
    reset_done = memory.get("reset_done", False)
    markers = _find_markers(obs)
    goals = _find_goals(obs, {m["color"] for m in markers})
    groups = _make_groups(obs, markers)
    pieces = _find_pieces(obs, groups, markers)

    selected = next(
        (i for i, marker in enumerate(markers) if marker["selected"]),
        0 if markers else None,
    )

    memory.clear()
    memory.update({
        "initialized": True,
        "reset_done": reset_done,
        "turn": 0,
        "markers": markers,
        "goals": goals,
        "groups": groups,
        "pieces": pieces,
        "selected": selected,
        "pending": None,
        "last_move": None,
        "seen": {},
        "bad_ball_positions": set(),
        "hazards": np.array(obs == 10, dtype=bool),
        "walls": np.array(obs == 2, dtype=bool),
    })


def _marker_strength(obs, x, y, color):
    if not (2 <= x <= 61 and 2 <= y <= 61):
        return 0
    if int(obs[y, x]) != color:
        return 0
    neighbors = [
        int(obs[y + dy, x + dx])
        for dx, dy in _CROSS
        if dx or dy
    ]
    return max(neighbors.count(0), neighbors.count(3))


def _sync_last_move(obs, memory):
    last = memory.get("last_move")
    memory["last_move"] = None
    if last is None:
        return

    marker_id, old_x, old_y, new_x, new_y, ball_x, ball_y = last
    markers = memory["markers"]
    if not (0 <= marker_id < len(markers)):
        return

    marker = markers[marker_id]
    color = marker["color"]
    old_strength = _marker_strength(obs, old_x, old_y, color)
    new_strength = _marker_strength(obs, new_x, new_y, color)

    # The ball can hide part of a marker, so require clear evidence
    # at the old position before declaring the move reversed.
    if old_strength >= 9 and new_strength < 9:
        marker["x"] = old_x
        marker["y"] = old_y
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if max(abs(dx), abs(dy)) <= 2:
                    memory["bad_ball_positions"].add(
                        (ball_x + dx, ball_y + dy)
                    )


def _make_maps(obs, memory):
    memory["hazards"] |= (obs == 10)
    memory["walls"] |= (obs == 2)

    walls = memory["walls"]
    hazards = memory["hazards"].copy()
    lines = obs == 1
    adjacent = hazards.copy()

    # Lines are drawn over both ordinary background and hazards.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys = slice(max(0, dy), min(64, 64 + dy))
            xs = slice(max(0, dx), min(64, 64 + dx))
            yt = slice(max(0, -dy), min(64, 64 - dy))
            xt = slice(max(0, -dx), min(64, 64 - dx))
            adjacent[yt, xt] |= hazards[ys, xs]

    hazards |= lines & adjacent

    marker_ok = np.zeros((64, 64), dtype=bool)
    marker_ok[2:62, 2:62] = True
    for dx, dy in _CROSS:
        marker_ok[2:62, 2:62] &= ~walls[
            2 + dy:62 + dy, 2 + dx:62 + dx
        ]

    ball_ok = np.zeros((64, 64), dtype=bool)
    ball_ok[2:62, 2:62] = True
    for dx, dy in _CIRCLE:
        ball_ok[2:62, 2:62] &= ~hazards[
            2 + dy:62 + dy, 2 + dx:62 + dx
        ]

    for x, y in memory["bad_ball_positions"]:
        if 0 <= x < 64 and 0 <= y < 64:
            ball_ok[y, x] = False

    return marker_ok, ball_ok


def _distances(target, goal, ball_ok):
    offsets = _GOAL_HITS if goal else _PIECE_HITS
    dist = [[9999] * 64 for _ in range(64)]
    queue = deque()

    # Seed only safe ball positions that actually overlap the objective.
    for dx, dy in offsets:
        x = target["x"] + dx
        y = target["y"] + dy
        if 2 <= x <= 61 and 2 <= y <= 61 and ball_ok[y, x]:
            if dist[y][x] != 0:
                dist[y][x] = 0
                queue.append((x, y))

    # Moving an anchor is a teleport, so hazards between two ball
    # positions do not block the move.
    while queue:
        x, y = queue.popleft()
        d = dist[y][x] + 1
        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            xx, yy = x + dx, y + dy
            if 2 <= xx <= 61 and 2 <= yy <= 61 and dist[yy][xx] > d:
                dist[yy][xx] = d
                queue.append((xx, yy))

    return dist


def _selectable(marker_id, markers):
    marker = markers[marker_id]
    for j, other in enumerate(markers):
        if (
            j != marker_id
            and abs(marker["x"] - other["x"]) <= 2
            and abs(marker["y"] - other["y"]) <= 2
        ):
            return False
    return True


def _candidate_tasks(memory):
    markers = memory["markers"]
    goals = memory["goals"]
    groups = memory["groups"]
    pieces = memory["pieces"]

    claimed = set()
    for group in groups:
        if group["kind"] != 4 or not group["colors"]:
            continue

        matches = [
            j for j, goal in enumerate(goals)
            if len(goal["colors"]) > 1
            and group["colors"] <= goal["colors"]
        ]
        if len(matches) == 1:
            claimed.add(matches[0])

    tasks = []
    completed = []

    for gi, group in enumerate(groups):
        bx, by = _center(group, markers)

        if group["kind"] != 4:
            for goal in goals:
                if goal["colors"] != {group["kind"]}:
                    continue
                if _hit(bx, by, goal, True):
                    completed.append((gi, goal))
                else:
                    tasks.append((gi, goal, True, None))
            continue

        colors = group["colors"]
        for j, goal in enumerate(goals):
            required = goal["colors"]
            if len(required) < 2 or not colors <= required:
                continue
            if not colors and j in claimed:
                continue

            missing = required - colors
            if not missing:
                if _hit(bx, by, goal, True):
                    completed.append((gi, goal))
                else:
                    tasks.append((gi, goal, True, required))
            else:
                if not all(
                    any(piece["color"] == color for piece in pieces)
                    for color in missing
                ):
                    continue

                for piece in pieces:
                    if piece["color"] in missing:
                        tasks.append((gi, piece, False, required))

    return tasks, completed


def _plan(obs, memory):
    markers = memory["markers"]
    groups = memory["groups"]
    pieces = memory["pieces"]
    selected = memory["selected"]
    marker_ok, ball_ok = _make_maps(obs, memory)
    tasks, completed = _candidate_tasks(memory)

    if not tasks and completed:
        gi, goal = completed[0]
        allowed = goal["colors"] if groups[gi]["kind"] == 4 else None
        tasks = [(gi, goal, True, allowed)]

    best = None
    cache = {}

    for gi, target, is_goal, allowed_colors in tasks:
        group = groups[gi]
        ids = group["markers"]
        if not ids:
            continue

        bx, by = _center(group, markers)
        if not (2 <= bx <= 61 and 2 <= by <= 61):
            continue

        key = (target["x"], target["y"], is_goal)
        if key not in cache:
            cache[key] = _distances(target, is_goal, ball_ok)
        dist = cache[key]
        start_d = dist[by][bx]
        if start_d >= 9999:
            continue

        sx = sum(markers[i]["x"] for i in ids)
        sy = sum(markers[i]["y"] for i in ids)
        n = len(ids)

        for marker_id in ids:
            if marker_id != selected and not _selectable(marker_id, markers):
                continue

            marker = markers[marker_id]
            base_x = sx - marker["x"]
            base_y = sy - marker["y"]

            for y in range(2, 62):
                for x in range(2, 62):
                    if not marker_ok[y, x]:
                        continue

                    # Clicking an existing marker selects it instead.
                    if any(
                        abs(x - other["x"]) <= 2
                        and abs(y - other["y"]) <= 2
                        for other in markers
                    ):
                        continue

                    nx = (base_x + x) // n
                    ny = (base_y + y) // n
                    if not ball_ok[ny, nx]:
                        continue

                    end_d = dist[ny][nx]
                    if end_d >= 9999:
                        continue

                    if group["kind"] == 4 and allowed_colors is not None:
                        if any(
                            piece["color"] not in allowed_colors
                            and _hit(nx, ny, piece)
                            for piece in pieces
                        ):
                            continue

                    gain = start_d - end_d
                    state_key = (
                        gi,
                        tuple(
                            (x, y) if i == marker_id
                            else (markers[i]["x"], markers[i]["y"])
                            for i in ids
                        ),
                    )
                    visits = memory["seen"].get(state_key, 0)
                    switch_cost = 0 if marker_id == selected else 1

                    rank = (
                        0 if gain > 0 or end_d == 0 else 1,
                        end_d * n / 60.0
                        + 0.16 * switch_cost
                        + 0.7 * visits,
                        -gain,
                        switch_cost,
                        -(
                            abs(x - marker["x"])
                            + abs(y - marker["y"])
                        ),
                    )

                    if best is None or rank < best[0]:
                        best = (rank, marker_id, x, y, state_key)

    return best


def _commit_move(memory, marker_id, x, y):
    markers = memory["markers"]
    marker = markers[marker_id]
    old_x, old_y = marker["x"], marker["y"]
    marker["x"], marker["y"] = x, y
    ball_x, ball_y = x, y

    for group in memory["groups"]:
        if marker_id not in group["markers"]:
            continue

        ball_x, ball_y = _center(group, markers)

        # Pickups occur before hazard handling, even if a move reverses.
        if group["kind"] == 4:
            remaining = []
            for piece in memory["pieces"]:
                if _hit(ball_x, ball_y, piece):
                    group["colors"].add(piece["color"])
                else:
                    remaining.append(piece)
            memory["pieces"] = remaining
        break

    memory["last_move"] = (
        marker_id, old_x, old_y, x, y, ball_x, ball_y
    )
    return "ACTION6", int(y), int(x)


def _reset(memory):
    memory.clear()
    memory["reset_done"] = True
    return "RESET", None, None


def act(obs, available, memory):
    if "ACTION6" not in available:
        return "RESET", None, None

    if not memory.get("initialized"):
        _initialize(obs, memory)
    else:
        _sync_last_move(obs, memory)

    memory["turn"] += 1
    markers = memory["markers"]

    if not markers or not memory["groups"] or not memory["goals"]:
        return _reset(memory)

    # If a hazardous perturbed state has consumed many actions,
    # restarting once can be cheaper than continuing a loop.
    if (
        memory["turn"] >= 22
        and not memory.get("reset_done", False)
        and bool(memory["hazards"].any())
        and "RESET" in available
    ):
        return _reset(memory)

    pending = memory.get("pending")
    if pending is not None:
        memory["pending"] = None
        marker_id, x, y, state_key = pending
        if 0 <= marker_id < len(markers):
            memory["seen"][state_key] = memory["seen"].get(state_key, 0) + 1
            return _commit_move(memory, marker_id, x, y)

    plan = _plan(obs, memory)
    if plan is None:
        return _reset(memory)

    _, marker_id, x, y, state_key = plan

    if marker_id != memory["selected"]:
        memory["selected"] = marker_id
        memory["pending"] = (marker_id, x, y, state_key)
        marker = markers[marker_id]
        return "ACTION6", int(marker["y"]), int(marker["x"])

    memory["seen"][state_key] = memory["seen"].get(state_key, 0) + 1
    return _commit_move(memory, marker_id, x, y)

import itertools
import math
from collections import deque

import numpy as np


HANDLE = [
    (0, -2), (-1, -1), (0, -1), (1, -1),
    (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
    (-1, 1), (0, 1), (1, 1), (0, 2),
]

BALL = [
    (x, y)
    for y in range(-2, 3)
    for x in range(-2, 3)
    if not (abs(x) == 2 and abs(y) == 2)
]

GOAL = [
    (x, y)
    for y in range(-3, 4)
    for x in range(-3, 4)
    if (abs(y) == 3 and abs(x) <= 1)
    or (abs(y) == 2 and abs(x) <= 2)
    or (abs(y) <= 1 and abs(x) <= 3)
]

RING = [
    (-1, -3), (1, -3),
    (-2, -2), (2, -2),
    (-3, -1), (3, -1),
    (-3, 1), (3, 1),
    (-2, 2), (2, 2),
    (-1, 3), (1, 3),
]

PIGMENTS = {7, 8, 9, 11, 12, 13, 14, 15}
GRID_Y, GRID_X = np.indices((64, 64))
BALL_SET = set(BALL)


def _overlap_offsets(a, b):
    return {(ax - bx, ay - by) for ax, ay in a for bx, by in b}


BALL_GOAL = _overlap_offsets(GOAL, BALL)
BALL_BALL = _overlap_offsets(BALL, BALL)


def _collides(p, q, offsets):
    return (p[0] - q[0], p[1] - q[1]) in offsets


def _shifted_obstacles(mask, offsets):
    result = np.zeros((64, 64), dtype=bool)
    for dx, dy in offsets:
        x0, x1 = max(0, -dx), min(64, 64 - dx)
        y0, y1 = max(0, -dy), min(64, 64 - dy)
        if x0 < x1 and y0 < y1:
            result[y0:y1, x0:x1] |= mask[
                y0 + dy:y1 + dy, x0 + dx:x1 + dx
            ]
    return result


def _handles(frame):
    found = []
    for y in range(2, 62):
        for x in range(2, 62):
            colour = int(frame[y, x])
            if colour not in (4, 14, 15):
                continue
            values = [
                int(frame[y + dy, x + dx])
                for dx, dy in HANDLE
                if dx or dy
            ]
            zeros = sum(v == 0 for v in values)
            greys = sum(v == 3 for v in values)
            if zeros >= 11:
                found.append((x, y, colour, True))
            elif greys >= 11:
                found.append((x, y, colour, False))
    return found


def _coloured_balls(frame):
    balls = []
    ys, xs = np.nonzero(frame == 6)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if not (2 <= x < 62 and 2 <= y < 62):
            continue
        neighbours = [
            int(frame[y - 1, x]), int(frame[y + 1, x]),
            int(frame[y, x - 1]), int(frame[y, x + 1]),
        ]
        for colour in (14, 15):
            if neighbours.count(colour) >= 3:
                balls.append((x, y, colour))
                break
    return balls


def _partial_handle_score(frame, x, y, marker, balls):
    """Check a handle whose pixels may be covered by a ball or wall."""
    if not (2 <= x < 62 and 2 <= y < 62):
        return None

    best = None
    for stroke, selected in ((3, False), (0, True)):
        visible = 0
        valid = True
        for dx, dy in HANDLE:
            px, py = x + dx, y + dy
            expected = marker if (dx == 0 and dy == 0) else stroke
            value = int(frame[py, px])
            if value == expected:
                if dx or dy:
                    visible += 1
                continue

            covered = value == 2
            if not covered:
                for bx, by, colour in balls:
                    if ((px - bx, py - by) in BALL_SET and
                            value in (colour, 6)):
                        covered = True
                        break
            if not covered:
                valid = False
                break

        if valid and visible >= 3:
            candidate = (visible, selected)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def _infer_handles(frame, handles):
    """
    Recover a partially obscured coloured handle using the known ball
    centre and the centroid rule.
    """
    handles = list(handles)
    balls = _coloured_balls(frame)

    for bx, by, marker in balls:
        same = [h for h in handles if h[2] == marker]
        if not same:
            continue

        # No inference is needed if a visible subset already explains the ball.
        explained = False
        for size in range(2, min(len(same), 6) + 1):
            for subset in itertools.combinations(same, size):
                cx = sum(h[0] for h in subset) // size
                cy = sum(h[1] for h in subset) // size
                if (cx, cy) == (bx, by):
                    explained = True
                    break
            if explained:
                break
        if explained:
            continue

        best = None
        for size in range(1, min(len(same), 5) + 1):
            for subset in itertools.combinations(same, size):
                count = size + 1
                sx = sum(h[0] for h in subset)
                sy = sum(h[1] for h in subset)
                x0 = count * bx - sx
                y0 = count * by - sy

                for x in range(x0, x0 + count):
                    for y in range(y0, y0 + count):
                        if any(abs(x - h[0]) <= 2 and abs(y - h[1]) <= 2
                               for h in handles):
                            continue
                        evidence = _partial_handle_score(
                            frame, x, y, marker, balls
                        )
                        if evidence is None:
                            continue
                        visible, selected = evidence
                        score = 100 * size + visible
                        if best is None or score > best[0]:
                            best = (score, (x, y, marker, selected))

        if best is not None:
            handles.append(best[1])

    return handles


def _ball_quality(frame, x, y, marker):
    if not (2 <= x < 62 and 2 <= y < 62):
        return -100
    vals = [int(frame[y + dy, x + dx]) for dx, dy in BALL]
    if marker == 4:
        allowed = PIGMENTS | {0}
        count = sum(v in allowed for v in vals)
        return count if count >= 19 else -100
    if int(frame[y, x]) != 6:
        return -100
    count = sum(v == marker or v == 6 for v in vals)
    return count if count >= 19 else -100


def _groups(frame, handles):
    result = []
    for marker in (4, 14, 15):
        hs = [h for h in handles if h[2] == marker]
        n = len(hs)
        if n < 2:
            continue

        candidates = []
        for size in range(2, min(n, 6) + 1):
            for inds in itertools.combinations(range(n), size):
                x = sum(hs[i][0] for i in inds) // size
                y = sum(hs[i][1] for i in inds) // size
                quality = _ball_quality(frame, x, y, marker)
                if quality >= 0:
                    bits = sum(1 << i for i in inds)
                    candidates.append((bits, inds, (x, y), quality))

        best = [(-1e9, [])]

        def search(remaining, chosen, score):
            if not remaining:
                if score > best[0][0]:
                    best[0] = (score, chosen[:])
                return
            first = (remaining & -remaining).bit_length() - 1
            search(remaining & ~(1 << first), chosen, score - 30)
            for bits, inds, centre, quality in candidates:
                if bits & (1 << first) and bits & remaining == bits:
                    chosen.append((inds, centre))
                    search(
                        remaining ^ bits, chosen,
                        score + 100 * len(inds) + quality + len(inds)
                    )
                    chosen.pop()

        search((1 << n) - 1, [], 0)
        for inds, centre in best[0][1]:
            group_handles = [hs[i] for i in inds]
            if marker == 4:
                x, y = centre
                colours = {
                    int(frame[y + dy, x + dx])
                    for dx, dy in BALL
                    if int(frame[y + dy, x + dx]) in PIGMENTS
                }
            else:
                colours = {marker}
            result.append({
                "marker": marker,
                "handles": group_handles,
                "ball": centre,
                "colours": colours,
            })
    return result


def _goals(frame):
    left = [
        (-1, -3), (-2, -2), (-3, -1),
        (-3, 1), (-2, 2), (-1, 3),
    ]
    right = [
        (1, -3), (2, -2), (3, -1),
        (3, 1), (2, 2), (1, 3),
    ]
    goals = []
    for y in range(3, 61):
        for x in range(3, 61):
            lv = [int(frame[y + dy, x + dx]) for dx, dy in left]
            rv = [int(frame[y + dy, x + dx]) for dx, dy in right]
            lc = max(PIGMENTS, key=lambda c: lv.count(c))
            rc = max(PIGMENTS, key=lambda c: rv.count(c))
            if lv.count(lc) >= 5 and rv.count(rc) >= 5:
                goals.append((x, y, tuple(sorted({lc, rc}))))
    return goals


def _blink_goals(frame, groups):
    """Recognize black goal outlines around already-placed coloured balls."""
    found = []
    for group in groups:
        colour = group["marker"]
        if colour not in (14, 15):
            continue
        bx, by = group["ball"]
        for y in range(max(3, by - 5), min(60, by + 5) + 1):
            for x in range(max(3, bx - 5), min(60, bx + 5) + 1):
                if not _collides((bx, by), (x, y), BALL_GOAL):
                    continue
                black = 0
                valid = True
                for dx, dy in RING:
                    px, py = x + dx, y + dy
                    value = int(frame[py, px])
                    if value == 0:
                        black += 1
                    elif ((px - bx, py - by) in BALL_SET and
                          value in (colour, 6)):
                        pass
                    elif value == 2:
                        pass
                    else:
                        valid = False
                        break
                if valid and black >= 6:
                    found.append((x, y, (colour,)))
    return found


def _pickup_templates():
    pattern = np.array([
        [-1, 1, 0, 0, -1],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [-1, 1, 1, 0, -1],
    ])
    templates = []
    for complement in (False, True):
        p = pattern.copy()
        if complement:
            p = np.where(p < 0, -1, 1 - p)
        for k in range(4):
            templates.append(np.rot90(p, k))
    return templates


PICKUP_TEMPLATES = _pickup_templates()


def _pickups(frame, groups):
    found = set()
    ball_centres = [g["ball"] for g in groups]
    for colour in PIGMENTS:
        for template in PICKUP_TEMPLATES:
            matches = np.ones((60, 60), dtype=bool)
            for yy in range(5):
                for xx in range(5):
                    v = int(template[yy, xx])
                    if v < 0:
                        continue
                    expected = colour if v == 1 else 0
                    matches &= (
                        frame[yy:yy + 60, xx:xx + 60] == expected
                    )
            ys, xs = np.nonzero(matches)
            for top_y, top_x in zip(ys.tolist(), xs.tolist()):
                x, y = top_x + 2, top_y + 2
                if any(abs(x - bx) <= 3 and abs(y - by) <= 3
                       for bx, by in ball_centres):
                    continue
                found.add((x, y, colour))
    return list(found)


def _distance(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _assign(groups, goals, pickups):
    mixed = [set(g[2]) for g in goals if len(g[2]) > 1]
    available_colours = {p[2] for p in pickups}
    for group in groups:
        if group["marker"] == 4:
            available_colours |= group["colours"]

    filtered = []
    for goal in goals:
        colours = set(goal[2])
        if len(colours) == 1 and any(colours < m for m in mixed):
            if not any(g["marker"] in colours and g["marker"] != 4
                       for g in groups):
                continue
        if any(g["marker"] == 4 for g in groups):
            if len(colours) == 1 and not colours <= available_colours:
                if not any(g["marker"] in colours and g["marker"] != 4
                           for g in groups):
                    continue
        filtered.append(goal)

    choices = []
    for group in groups:
        options = []
        for j, goal in enumerate(filtered):
            target_colours = set(goal[2])
            if group["marker"] != 4:
                if target_colours != {group["marker"]}:
                    continue
                missing = set()
            else:
                if not group["colours"] <= target_colours:
                    continue
                missing = target_colours - group["colours"]
                if any(not any(p[2] == c for p in pickups)
                       for c in missing):
                    continue

            b = group["ball"]
            done = (
                not missing and
                _collides(b, (goal[0], goal[1]), BALL_GOAL)
            )
            if done:
                cost = 0.0
            else:
                pos = b
                cost = 0.0
                remaining = set(missing)
                while remaining:
                    candidates = [p for p in pickups if p[2] in remaining]
                    p = min(candidates, key=lambda p: _distance(pos, p[:2]))
                    cost += _distance(pos, p[:2])
                    pos = p[:2]
                    remaining.remove(p[2])
                cost += _distance(pos, goal[:2])
            options.append((j, cost, missing, done))
        choices.append(options)

    best = [float("inf"), None]

    def search(i, used, total, assignment):
        if total >= best[0]:
            return
        if i == len(groups):
            best[:] = [total, assignment[:]]
            return
        for option in choices[i]:
            j = option[0]
            if j not in used:
                used.add(j)
                assignment.append(option)
                search(i + 1, used, total + option[1], assignment)
                assignment.pop()
                used.remove(j)

    search(0, set(), 0.0, [])
    return filtered, best[1]


def _bfs_distances(blocked, target):
    tx, ty = target
    dist = np.full((64, 64), 10000, dtype=np.int32)
    starts = []
    if 0 <= tx < 64 and 0 <= ty < 64 and not blocked[ty, tx]:
        starts.append((tx, ty))
    else:
        for radius in range(1, 6):
            for y in range(max(0, ty - radius), min(64, ty + radius + 1)):
                for x in range(max(0, tx - radius), min(64, tx + radius + 1)):
                    if (max(abs(x - tx), abs(y - ty)) == radius and
                            not blocked[y, x]):
                        starts.append((x, y))
            if starts:
                break

    queue = deque()
    for x, y in starts:
        dist[y, x] = 0
        queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        nd = int(dist[y, x]) + 1
        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ):
            nx, ny = x + dx, y + dy
            if (0 <= nx < 64 and 0 <= ny < 64 and
                    not blocked[ny, nx] and dist[ny, nx] > nd):
                dist[ny, nx] = nd
                queue.append((nx, ny))
    return dist


def _choose_move(frame, group, target, target_colours, pickups,
                 all_handles, hazard, failed):
    wall = frame == 2
    wall[:, 0] = False
    invalid_handle = _shifted_obstacles(wall, HANDLE)

    invalid_handle[:2, :] = True
    invalid_handle[62:, :] = True
    invalid_handle[:, :2] = True
    invalid_handle[:, 62:] = True

    if np.count_nonzero(hazard) > 50:
        bad_ball = _shifted_obstacles(hazard | wall, BALL)
    else:
        bad_ball = np.zeros((64, 64), dtype=bool)

    if group["marker"] == 4:
        for px, py, colour in pickups:
            if colour not in target_colours:
                for dx, dy in BALL_BALL:
                    x, y = px + dx, py + dy
                    if 0 <= x < 64 and 0 <= y < 64:
                        bad_ball[y, x] = True

    for x, y in failed:
        if 0 <= x < 64 and 0 <= y < 64:
            bad_ball[y, x] = True

    distances = _bfs_distances(bad_ball, target)
    hs = group["handles"]
    n = len(hs)
    sum_x = sum(h[0] for h in hs)
    sum_y = sum(h[1] for h in hs)
    best_score = float("inf")
    best = None

    for i, h in enumerate(hs):
        hx, hy, _, selected = h
        bx = (sum_x - hx + GRID_X) // n
        by = (sum_y - hy + GRID_Y) // n

        valid = ~invalid_handle.copy()
        valid &= ~bad_ball[by, bx]

        for other in all_handles:
            ox, oy = other[0], other[1]
            valid &= ~(
                (np.abs(GRID_X - ox) <= 2) &
                (np.abs(GRID_Y - oy) <= 2)
            )
            if (ox, oy) != (hx, hy):
                valid &= ~(
                    (np.abs(GRID_X - ox) <= 3) &
                    (np.abs(GRID_Y - oy) <= 3)
                )

        valid &= ~(
            (np.abs(GRID_X - bx) <= 2) &
            (np.abs(GRID_Y - by) <= 2)
        )

        d = distances[by, bx].astype(float)
        euclid = np.hypot(bx - target[0], by - target[1])
        d = np.where(d >= 10000, 200.0 + euclid, d + 0.08 * euclid)
        movement = np.hypot(GRID_X - hx, GRID_Y - hy)
        score = d + (0.7 if not selected else 0.0) + 0.001 * movement
        score[~valid] = 1e9
        index = int(np.argmin(score))
        value = float(score.flat[index])
        if value < best_score:
            yy, xx = np.unravel_index(index, score.shape)
            best_score = value
            best = (
                i, int(xx), int(yy),
                int(bx[yy, xx]), int(by[yy, xx])
            )
    return best


def act(obs, available, memory):
    frame = np.asarray(obs)
    handles = _infer_handles(frame, _handles(frame))
    groups = _groups(frame, handles)

    if not groups:
        if "RESET" in available and not memory.get("reset_tried"):
            memory["reset_tried"] = True
            return ("RESET", None, None)
        return ("ACTION6", 32, 32)

    known_goals = memory.setdefault("goals", [])
    for goal in _goals(frame) + _blink_goals(frame, groups):
        if not any(abs(goal[0] - g[0]) <= 1 and
                   abs(goal[1] - g[1]) <= 1 for g in known_goals):
            known_goals.append(goal)

    pickups = _pickups(frame, groups)

    current_hazard = frame == 10
    if np.count_nonzero(current_hazard) > 50:
        if "hazard" in memory:
            memory["hazard"] = (
                np.asarray(memory["hazard"], dtype=bool) | current_hazard
            )
        else:
            memory["hazard"] = current_hazard.copy()
    hazard = memory.get("hazard", np.zeros((64, 64), dtype=bool))

    failed = memory.setdefault("failed", [])
    last = memory.pop("last_move", None)
    if last is not None:
        old, new, predicted = last
        old_present = any(
            h[0] == old[0] and h[1] == old[1] for h in handles
        )
        new_present = any(
            h[0] == new[0] and h[1] == new[1] for h in handles
        )
        if old_present and not new_present and predicted not in failed:
            failed.append(predicted)

    goals, assignment = _assign(groups, known_goals, pickups)
    if assignment is None:
        if "RESET" in available and not memory.get("reset_tried"):
            memory["reset_tried"] = True
            return ("RESET", None, None)
        if known_goals:
            active = next((h for h in handles if h[3]), handles[0])
            goal = min(
                known_goals,
                key=lambda g: _distance(active[:2], g[:2])
            )
            return ("ACTION6", int(goal[1]), int(goal[0]))
        return ("ACTION6", 32, 32)

    tasks = []
    for i, (goal_index, cost, missing, done) in enumerate(assignment):
        if done:
            continue
        group = groups[i]
        goal = goals[goal_index]
        target_colours = set(goal[2])
        if missing:
            candidates = [p for p in pickups if p[2] in missing]
            if not candidates:
                continue
            pickup = min(
                candidates, key=lambda p: _distance(group["ball"], p[:2])
            )
            target = pickup[:2]
        else:
            target = goal[:2]
        selected = any(h[3] for h in group["handles"])
        priority = _distance(group["ball"], target)
        if selected:
            priority -= 3.0
        tasks.append((priority, i, target, target_colours))

    if not tasks:
        return ("ACTION6", 32, 32)

    tasks.sort(key=lambda item: item[0])
    for _, i, target, target_colours in tasks:
        group = groups[i]
        move = _choose_move(
            frame, group, target, target_colours, pickups,
            handles, hazard, failed
        )
        if move is None:
            continue
        hi, x, y, bx, by = move
        handle = group["handles"][hi]
        if not handle[3]:
            return ("ACTION6", int(handle[1]), int(handle[0]))
        memory["last_move"] = (
            (handle[0], handle[1]), (x, y), (bx, by)
        )
        return ("ACTION6", y, x)

    if "RESET" in available and not memory.get("reset_tried"):
        memory["reset_tried"] = True
        return ("RESET", None, None)
    return ("ACTION6", 32, 32)

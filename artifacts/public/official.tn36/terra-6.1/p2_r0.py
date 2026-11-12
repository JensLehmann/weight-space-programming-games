import heapq
import time
import numpy as np


def _rotate(a, rotation):
    return np.rot90(a, -(rotation // 90))


def _find_panel(obs):
    candidates = []
    for y in range(24, 50):
        x = 0
        while x < 64:
            if obs[y, x] != 0:
                x += 1
                continue
            start = x
            while x < 64 and obs[y, x] == 0:
                x += 1
            end = x - 1
            if end - start + 1 < 15:
                continue

            columns = []
            for c in range(max(start + 2, 2), min(end - 1, 62)):
                bits = []
                clicks = []
                for bit in range(6):
                    if bit % 2 == 0:
                        yy = y + 1 + 6 * (bit // 2)
                        if yy >= 51:
                            break
                        v = int(obs[yy, c])
                        if (
                            v not in (1, 5)
                            or any(int(obs[yy, xx]) != v
                                   for xx in (c - 1, c + 1))
                            or int(obs[yy, c - 2]) != 0
                            or int(obs[yy, c + 2]) != 0
                        ):
                            break
                        clicks.append((yy, c))
                    else:
                        yy = y + 3 + 6 * (bit // 2)
                        if yy + 2 >= 51:
                            break
                        v = int(obs[yy, c])
                        if (
                            v not in (1, 5)
                            or any(int(obs[yy + d, c]) != v
                                   for d in (1, 2))
                            or any(
                                int(obs[yy + d, c - 1]) != 0
                                or int(obs[yy + d, c + 1]) != 0
                                for d in (0, 1, 2)
                            )
                        ):
                            break
                        clicks.append((yy + 1, c))
                    bits.append(v == 5)

                if len(bits) >= 2 and len(bits) % 2 == 0:
                    columns.append((c, bits, clicks))

            for first in range(len(columns)):
                group = [columns[first]]
                for item in columns[first + 1:]:
                    if item[0] == group[-1][0] + 5:
                        group.append(item)
                    elif item[0] > group[-1][0] + 5:
                        break
                if len(group) < 2:
                    continue
                nbits = min(len(item[1]) for item in group)
                if nbits not in (2, 4, 6):
                    continue
                values = []
                click_positions = []
                for _, bits, clicks in group:
                    value = sum(
                        (1 << i) for i in range(nbits) if bits[i]
                    )
                    values.append(value)
                    click_positions.append(clicks[:nbits])
                candidates.append(
                    (y, -len(group), start, end, values, click_positions)
                )

    if not candidates:
        return None
    candidates.sort()
    y, _, left, right, values, clicks = candidates[0]
    return y, left, right, values, clicks


def _template_matches(obs, left, right, bottom, template, scales):
    region = obs[:bottom, left:right + 1]
    height, width = region.shape
    results = []

    for scale in scales:
        for rotation in (0, 90, 180, 270):
            mask = _rotate(template, rotation)
            mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
            h, w = mask.shape
            if h > height or w > width:
                continue

            possible = np.ones(
                (height - h + 1, width - w + 1), dtype=bool
            )
            for dy, dx in zip(*np.nonzero(mask)):
                possible &= (
                    region[
                        dy:dy + height - h + 1,
                        dx:dx + width - w + 1
                    ] == 11
                )
                if not possible.any():
                    break

            ys, xs = np.nonzero(possible)
            for yy, xx in zip(ys.tolist(), xs.tolist()):
                patch = region[yy:yy + h, xx:xx + w]
                extra = int(np.count_nonzero((patch == 11) & ~mask))
                results.append((
                    extra / float(scale * scale),
                    -scale,
                    left + xx,
                    yy,
                    rotation,
                    scale
                ))
    return results


def _find_pieces(obs, panel):
    py, left, right, _, _ = panel

    goal_template = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ], dtype=bool)

    actor_template = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
    ], dtype=bool)

    goals = _template_matches(
        obs, left, right, py, goal_template, (1, 2, 3, 4)
    )
    if not goals:
        return None
    goals.sort()
    _, _, gx, gy, gr, gs = goals[0]
    goal = (gx + gs, gy + gs, gr, gs)

    actors = _template_matches(
        obs, left, right, py, actor_template, (1, 2, 3, 4)
    )
    if not actors:
        return None

    def actor_key(item):
        extra, negscale, x, y, rotation, scale = item
        overlap = not (
            x + 4 * scale <= gx
            or gx + 6 * gs <= x
            or y + 4 * scale <= gy
            or gy + 6 * gs <= y
        )
        return (
            extra + (3.0 if overlap else 0.0),
            negscale,
            y,
            x
        )

    actors.sort(key=actor_key)
    _, _, ax, ay, ar, actor_scale = actors[0]
    return (ax, ay, ar, actor_scale), goal


def _find_run_button(obs):
    offsets = []
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if abs(dy) == 4:
                inside = abs(dx) <= 2
            elif abs(dy) == 3:
                inside = abs(dx) <= 3
            else:
                inside = abs(dx) <= 4
            if inside:
                offsets.append((dy, dx))

    for cy in range(52, 60):
        for cx in range(4, 60):
            if all(
                int(obs[cy + dy, cx + dx]) == 9
                for dy, dx in offsets
            ):
                return cy, cx
    return None


def _find_checkpoints(obs, panel, actor):
    py, left, right, _, _ = panel
    ax, ay, _, _ = actor
    checkpoints = set()

    for y in range(0, py - 3):
        if (y - ay) % 4:
            continue
        for x in range(left, right - 2):
            if (x - ax) % 4:
                continue
            patch = obs[y:y + 4, x:x + 4]
            for parity in (0, 1):
                good = True
                for dy in range(4):
                    for dx in range(4):
                        v = int(patch[dy, dx])
                        if (dx + dy) % 2 == parity:
                            if v != 11:
                                good = False
                                break
                        elif v == 11:
                            good = False
                            break
                    if not good:
                        break
                if good:
                    checkpoints.add((x, y))
                    break
    return checkpoints


def _search(obs, panel, actor, goal, checkpoints, forbidden, deadline):
    py, left, right, current, positions = panel
    n = len(current)
    nbits = len(positions[0])
    max_code = (1 << nbits) - 1
    max_scale = max(actor[3], goal[3])

    walls = {
        (x, y)
        for y in range(py)
        for x in range(left, right + 1)
        if int(obs[y, x]) == 6
    }
    blocked_cache = {}

    def blocked(x, y, scale):
        key = (x, y, scale)
        if key not in blocked_cache:
            size = 4 * scale
            blocked_cache[key] = any(
                (xx, yy) in walls
                for yy in range(y, y + size)
                for xx in range(x, x + size)
            )
        return blocked_cache[key]

    groups = [
        [1, 34], [2], [3], [33],
        [5], [6, 16], [7], [8], [9],
        [10, 11], [12, 13],
    ]
    implemented = {
        0, 1, 2, 3, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 33, 34, 63
    }

    options = []
    for old in current:
        choices = []
        noops = [
            v for v in range(max_code + 1)
            if v == 0 or v not in implemented
        ]
        if noops:
            v = min(noops, key=lambda z: ((z ^ old).bit_count(), z))
            choices.append((v, 0))
        for group in groups:
            valid = [v for v in group if v <= max_code]
            if valid:
                v = min(valid, key=lambda z: ((z ^ old).bit_count(), z))
                choices.append((v, group[0]))
        options.append(choices)

    xmin, xmax = left - 12, right + 12
    ymin, ymax = -12, py + 12

    start = (
        actor[0], actor[1], actor[2], actor[3],
        0, 0, 1 if forbidden and len(forbidden) == n else 0
    )
    dist = {start: 0}
    parent = {}
    heap = [(0, start)]
    finish = None
    expansions = 0

    while heap:
        cost, state = heapq.heappop(heap)
        if cost != dist.get(state):
            continue

        expansions += 1
        if expansions % 2048 == 0 and time.monotonic() > deadline:
            break

        x, y, rotation, scale, slot, stage, matching = state

        if slot == n:
            if (x, y, rotation, scale) == goal:
                if not (stage == 0 and matching):
                    finish = state
                    break

            if (x, y) in checkpoints and scale == 1:
                if not (stage == 0 and matching):
                    nxt = (x, y, rotation, scale, 0, 1, 0)
                    nc = cost + 1
                    if nc < dist.get(nxt, 10**9):
                        dist[nxt] = nc
                        parent[nxt] = (state, None)
                        heapq.heappush(heap, (nc, nxt))
            continue

        for code, effect in options[slot]:
            nx, ny, nr, ns = x, y, rotation, scale

            if effect == 1:
                nx -= 4
            elif effect == 2:
                nx += 4
            elif effect == 3:
                ny += 4
            elif effect == 33:
                ny -= 4
            elif effect == 10:
                nx += 8
            elif effect == 12:
                nx -= 8
            elif effect == 5:
                nr = (nr + 90) % 360
            elif effect == 6:
                nr = (nr - 90) % 360
            elif effect == 7:
                nr = (nr + 180) % 360
            elif effect == 8:
                ns += 1
            elif effect == 9:
                ns = max(1, ns - 1)

            if ns > max_scale:
                continue
            if not (xmin <= nx <= xmax and ymin <= ny <= ymax):
                continue

            if (nx != x or ny != y or ns != scale) and blocked(nx, ny, ns):
                nx, ny, ns = x, y, scale

            nm = matching
            if stage == 0 and nm:
                nm = int(code == forbidden[slot])

            nxt = (nx, ny, nr, ns, slot + 1, stage, nm)
            nc = cost + (code ^ current[slot]).bit_count()
            if nc < dist.get(nxt, 10**9):
                dist[nxt] = nc
                parent[nxt] = (state, code)
                heapq.heappush(heap, (nc, nxt))

    if finish is None:
        return None

    path = []
    state = finish
    while state != start:
        prev, code = parent[state]
        path.append((state, code))
        state = prev
    path.reverse()

    program = []
    expected = None
    for state, code in path:
        if code is None:
            break
        program.append(code)
        if len(program) == n:
            expected = list(state[:4])
            break

    if len(program) != n:
        return None
    return program, expected


def act(obs, available, memory):
    obs = np.asarray(obs)

    queue = memory.get("queue")
    if queue:
        row, col = queue.pop(0)
        if not queue:
            memory["pending"] = memory.pop("runmeta", None)
        return "ACTION6", int(row), int(col)

    if "ACTION6" not in available:
        if "RESET" in available:
            return "RESET", None, None
        return available[0], None, None

    panel = _find_panel(obs)
    pieces = _find_pieces(obs, panel) if panel is not None else None
    button = _find_run_button(obs)

    if panel is None or pieces is None or button is None:
        if "RESET" in available and not memory.get("parser_reset"):
            memory["parser_reset"] = True
            return "RESET", None, None
        return "ACTION6", 63, 63

    actor, goal = pieces
    pending = memory.pop("pending", None)
    if pending:
        if (
            tuple(pending["start"]) == tuple(actor)
            and tuple(pending["expected"]) != tuple(actor)
        ):
            memory["failed_start"] = list(actor)
            memory["failed_program"] = pending["program"]
        else:
            memory.pop("failed_start", None)
            memory.pop("failed_program", None)

    forbidden = None
    if memory.get("failed_start") == list(actor):
        forbidden = memory.get("failed_program")

    checkpoints = _find_checkpoints(obs, panel, actor)
    result = _search(
        obs, panel, actor, goal, checkpoints, forbidden,
        time.monotonic() + 8.0
    )

    if result is None:
        if "RESET" in available and not memory.get("search_reset"):
            memory["search_reset"] = True
            memory.pop("failed_start", None)
            memory.pop("failed_program", None)
            return "RESET", None, None
        return "ACTION6", int(button[0]), int(button[1])

    program, expected = result
    current = panel[3]
    positions = panel[4]
    actions = []

    for i, (old, new) in enumerate(zip(current, program)):
        difference = old ^ new
        for bit, position in enumerate(positions[i]):
            if difference & (1 << bit):
                actions.append([int(position[0]), int(position[1])])
    actions.append([int(button[0]), int(button[1])])

    memory["queue"] = actions
    memory["runmeta"] = {
        "start": list(actor),
        "expected": expected,
        "program": program,
    }

    row, col = memory["queue"].pop(0)
    if not memory["queue"]:
        memory["pending"] = memory.pop("runmeta")
    return "ACTION6", int(row), int(col)

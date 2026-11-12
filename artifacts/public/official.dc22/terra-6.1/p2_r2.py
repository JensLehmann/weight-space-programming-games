from collections import deque
import numpy as np


def _player(obs):
    ys, xs = np.where(obs == 14)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min())


def _layout(obs):
    y0 = 0
    while y0 < 62 and np.all(obs[y0] == 3):
        y0 += 1
    height = max(1, 63 - y0)
    panel = 38
    for x in range(20, 62):
        if np.count_nonzero(obs[y0:63, x] == 5) > height * 0.55:
            panel = max(2, x - 2)
            break
    return y0, panel - panel % 2


def _controls(obs, panel):
    found = {}
    for colour in (8, 9, 6, 15):
        for y in range(59):
            done = False
            for x in range(panel, 52):
                if (np.all(obs[y:y + 3, x + 3:x + 10] == colour)
                        and np.all(obs[y + 3:y + 5, x:x + 13] == colour)):
                    found[colour] = (y + 1, x + 6)
                    done = True
                    break
            if done:
                break

    for y in range(58):
        done = False
        for x in range(panel, 59):
            if np.all(obs[y:y + 6, x:x + 6] == 2):
                found[2] = (y + 2, x + 2)
                done = True
                break
        if done:
            break

    for y in range(57):
        done = False
        for x in range(panel, 58):
            if (np.all(obs[y:y + 3, x:x + 4] == 11)
                    and np.all(obs[y + 3:y + 7, x:x + 7] == 11)):
                found[11] = (y + 4, x + 2)
                done = True
                break
        if done:
            break
    return found


def _tiles(obs, y0, panel):
    cols = panel // 2
    rows = (63 - y0) // 2
    result = []
    for r in range(rows):
        y = y0 + 2 * r
        for c in range(cols):
            x = 2 * c
            result.append(
                int(obs[y, x])
                | (int(obs[y, x + 1]) << 4)
                | (int(obs[y + 1, x]) << 8)
                | (int(obs[y + 1, x + 1]) << 12)
            )
    return result, rows, cols


def _colours(code):
    return (code & 15, (code >> 4) & 15,
            (code >> 8) & 15, (code >> 12) & 15)


def _has_player(code):
    return 14 in _colours(code)


def _status(code, dynamic=False):
    a = _colours(code)
    if 14 in a:
        return False, False

    counts = [0] * 16
    for v in a:
        counts[v] += 1

    blocked = counts[0] > 0
    safe = counts[2] > 0 or counts[11] == 4

    if counts[6] and counts[7]:
        safe = True

    for colour in (9, 15):
        n = counts[colour]
        if n == 4:
            safe = True
        elif n == 2 and all(v in (colour, 2, 4, 5) for v in a):
            blocked = True
        elif colour == 15 and n == 3 and counts[2]:
            safe = True

    if counts[1] or counts[12] or counts[13]:
        safe = True

    if counts[8]:
        if dynamic:
            safe = True
        else:
            blocked = True

    return safe, blocked


def _index(x, y, y0, rows, cols):
    c = x // 2
    r = (y - y0) // 2
    if 0 <= r < rows and 0 <= c < cols:
        return r * cols + c
    return None


def _direction(a, b, cols):
    d = b - a
    if d == -cols:
        return "ACTION1"
    if d == cols:
        return "ACTION2"
    if d == -1:
        return "ACTION3"
    if d == 1:
        return "ACTION4"
    return None


def _neighbours(i, rows, cols):
    r, c = divmod(i, cols)
    if r > 0:
        yield i - cols
    if r + 1 < rows:
        yield i + cols
    if c > 0:
        yield i - 1
    if c + 1 < cols:
        yield i + 1


def _walk_to_orange(start, board, rows, cols):
    if start is None:
        return None
    q = deque([start])
    parent = {start: None}
    target = None

    while q:
        i = q.popleft()
        if i != start and 2 in _colours(board[i]):
            safe, blocked = _status(board[i])
            if safe and not blocked:
                target = i
                break
        for j in _neighbours(i, rows, cols):
            if j in parent:
                continue
            safe, blocked = _status(board[j], dynamic=True)
            if safe and not blocked:
                parent[j] = i
                q.append(j)

    if target is None:
        return None
    while parent[target] != start:
        target = parent[target]
    return _direction(start, target, cols)


def _update_observations(memory, board):
    history = memory.setdefault("history", {})
    foundation = memory.setdefault("foundation", set())
    goals = memory.setdefault("goals", set())
    teleporters = memory.setdefault("teleporters", set())
    keys = set()

    for i, code in enumerate(board):
        a = _colours(code)
        if 14 not in a:
            history[i] = code
        if 2 in a:
            foundation.add(i)
        if all(v == 11 for v in a):
            goals.add(i)
            foundation.add(i)
        if 6 in a and 7 in a:
            teleporters.add(i)
            foundation.add(i)
        if a.count(15) == 3:
            keys.add(i)
    memory["keys"] = keys


def _detect_crane(obs, y0, panel):
    buttons = None
    for y in range(6, 60):
        if buttons is not None:
            break
        for x in range(max(panel + 5, 5), 51):
            if (np.all(obs[y:y + 3, x:x + 9] == 8)
                    and np.all(obs[y - 6:y - 2, x - 5:x - 1] == 10)
                    and np.all(obs[y - 6:y - 2, x:x + 4] == 9)
                    and np.all(obs[y - 6:y - 2, x + 5:x + 9] == 10)
                    and np.all(obs[y - 6:y - 2, x + 10:x + 14] == 9)):
                buttons = {
                    "left": (y - 5, x - 4),
                    "up": (y - 5, x + 1),
                    "right": (y - 5, x + 6),
                    "down": (y - 5, x + 11),
                    "grip": (y + 1, x + 4),
                }
                break

    if buttons is None:
        return None

    object_pos = None
    for y in range(y0, 52):
        if object_pos is not None:
            break
        for x in range(2, max(2, panel - 3)):
            if np.all(obs[y:y + 12, x:x + 4] == 12):
                object_pos = (x - 2, y)
                break
    if object_pos is None:
        return None

    patterns = [
        [
            "...#####",
            "...#...#",
            ".####...",
            "####....",
            "####....",
            ".####...",
            "...#...#",
            "...#####",
        ],
        [
            "........",
            "...#####",
            ".####..#",
            "####....",
            "####....",
            ".####..#",
            "...#####",
            "........",
        ],
    ]
    points = [
        [(dx, dy) for dy, row in enumerate(pattern)
         for dx, ch in enumerate(row) if ch == "#"]
        for pattern in patterns
    ]

    crane_pos = None
    pattern_id = 0
    for y in range(y0, 56):
        if crane_pos is not None:
            break
        for x in range(max(0, panel - 7)):
            matched = False
            for k, pts in enumerate(points):
                if all(obs[y + dy, x + dx] == 8 for dx, dy in pts):
                    crane_pos = (x, y)
                    pattern_id = k
                    matched = True
                    break
            if matched:
                break

    if crane_pos is None:
        return None

    px, py = crane_pos
    ox, oy = object_pos

    # Infer the track origin from the initial platform position. This also
    # handles a crane already moved a few steps at episode start.
    bx = ox - 18
    by = oy + 14
    if ((px - bx) % 4 == 0 and (by - py) % 4 == 0):
        cx = (px - bx) // 4
        cy = (by - py) // 4
    else:
        cx = cy = -1

    if not (0 <= cx <= 3 and 0 <= cy <= 3
            and (cx == 0 or cy in (0, 3))):
        bx, by = px, py
        cx, cy = 0, 0

    dx = ox + 3 - (bx + 9)
    dy = (by + 4) - (oy + 6)
    if dx % 4 or dy % 4:
        return None
    gx, gy = dx // 4, dy // 4
    if not (0 <= gx <= 3 and 0 <= gy <= 3):
        return None

    attached = (
        pattern_id == 1
        and px + 9 == ox + 3
        and py + 4 == oy + 6
    )

    return {
        "base_x": bx,
        "base_y": by,
        "object_x": ox,
        "object_y": oy,
        "grab_x": gx,
        "grab_y": gy,
        "cx": cx,
        "cy": cy,
        "attached": attached,
        "buttons": buttons,
        "stage": "prepare",
    }


def _crane_route(start, target):
    if start == target:
        return []
    q = deque([start])
    parent = {start: None}

    while q:
        x, y = q.popleft()
        moves = []
        if x == 0 and y < 3:
            moves.append(("up", (x, y + 1)))
        if x == 0 and y > 0:
            moves.append(("down", (x, y - 1)))
        if y in (0, 3) and x > 0:
            moves.append(("left", (x - 1, y)))
        if y in (0, 3) and x < 3:
            moves.append(("right", (x + 1, y)))

        for name, nxt in moves:
            if nxt in parent:
                continue
            parent[nxt] = ((x, y), name)
            if nxt == target:
                path = []
                cur = nxt
                while parent[cur] is not None:
                    prev, action = parent[cur]
                    path.append(action)
                    cur = prev
                return path[::-1]
            q.append(nxt)
    return None


def _crane_action(crane, target=None, grip=False):
    if grip:
        name = "grip"
        crane["attached"] = True
    else:
        route = _crane_route((crane["cx"], crane["cy"]), target)
        if not route:
            return None
        name = route[0]
        if name == "up":
            crane["cy"] += 1
        elif name == "down":
            crane["cy"] -= 1
        elif name == "left":
            crane["cx"] -= 1
        elif name == "right":
            crane["cx"] += 1
    return crane["buttons"][name]


def _make_model(memory, board):
    n = len(board)
    groups = memory.get("groups", {})
    group_ids = sorted(groups)
    samples = [groups[k]["samples"] for k in group_ids]

    variable = set()
    group_variable = []
    for ss in samples:
        changed = []
        if len(ss) > 1:
            for i in range(n):
                values = {s[i] for s in ss if not _has_player(s[i])}
                if len(values) > 1:
                    changed.append(i)
                    variable.add(i)
        group_variable.append(changed)

    foundation = set(memory.get("foundation", ()))
    for ss in samples:
        for s in ss:
            for i, code in enumerate(s):
                if not _has_player(code) and 2 in _colours(code):
                    foundation.add(i)

    fixed_safe = 0
    fixed_block = 0
    history = memory.get("history", {})
    for i in range(n):
        if i in foundation:
            fixed_safe |= 1 << i
        if i in variable:
            continue
        code = board[i]
        if _has_player(code):
            code = history.get(i, code)
        safe, blocked = _status(code)
        if safe:
            fixed_safe |= 1 << i
        if blocked:
            fixed_block |= 1 << i

    phase_masks = []
    for group_id, ss, changed in zip(group_ids, samples, group_variable):
        masks = []
        for s in ss:
            safe_bits = 0
            block_bits = 0
            for i in changed:
                code = s[i]
                safe, blocked = _status(code, dynamic=True)
                colours = _colours(code)

                if group_id == 2 and 1 in colours and 12 not in colours:
                    safe = False
                    blocked = True

                if safe:
                    safe_bits |= 1 << i
                if blocked:
                    block_bits |= 1 << i
            masks.append((safe_bits, block_bits))
        phase_masks.append(masks)

    phases = tuple(
        groups[k].get("phase", 0) % len(groups[k]["samples"])
        for k in group_ids
    )
    return group_ids, phases, fixed_safe, fixed_block, phase_masks


def _search(memory, board, rows, cols, start, controls, extra_goals=None):
    if start is None:
        return None

    ids, initial_phases, fs, fb, masks = _make_model(memory, board)
    goals = set(memory.get("goals", ()))
    keys = set(memory.get("keys", ()))
    need_key = 15 not in controls and bool(keys)
    tele = sorted(memory.get("teleporters", ()))
    forbidden = memory.get("forbidden", set())

    if extra_goals is None:
        targets = goals | (keys if need_key else set())
    else:
        targets = set(extra_goals)

    cache = {}

    def surfaces(phases):
        if phases in cache:
            return cache[phases]
        safe, blocked = fs, fb
        for j, p in enumerate(phases):
            s, b = masks[j][p]
            safe |= s
            blocked |= b
        cache[phases] = (safe, blocked)
        return safe, blocked

    initial = (start, initial_phases)
    queue = deque([initial])
    parent = {initial: None}
    reached = None

    while queue and len(parent) < 180000:
        state = queue.popleft()
        pos, phases = state
        if pos in targets and state != initial:
            reached = state
            break

        safe, blocked = surfaces(phases)
        for nxt in _neighbours(pos, rows, cols):
            if not ((safe >> nxt) & 1) or ((blocked >> nxt) & 1):
                continue
            action = _direction(pos, nxt, cols)
            if (pos, phases, action) in forbidden:
                continue
            ns = (nxt, phases)
            if ns not in parent:
                parent[ns] = (state, (action, None))
                queue.append(ns)

        for j, key in enumerate(ids):
            if key not in controls:
                continue
            count = len(masks[j])
            if count <= 1:
                continue

            next_phases = list(phases)
            next_phases[j] = (next_phases[j] + 1) % count
            next_phases = tuple(next_phases)
            next_pos = pos

            if key == 6 and pos in tele and len(tele) >= 2:
                next_pos = next(t for t in tele if t != pos)

            new_safe, _ = surfaces(next_phases)
            if not ((new_safe >> next_pos) & 1):
                continue

            ns = (next_pos, next_phases)
            if ns not in parent:
                parent[ns] = (state, ("ACTION6", key))
                queue.append(ns)

    if reached is None:
        return None
    while parent[reached] is not None and parent[reached][0] != initial:
        reached = parent[reached][0]
    if parent[reached] is None:
        return None
    return parent[reached][1]


def _finish_sampling(memory, key, samples, board, force_eight):
    if force_eight:
        n = len(samples)
        period = n
        for p in range(1, n + 1):
            if n % p == 0 and all(
                    samples[i] == samples[i % p] for i in range(n)):
                period = p
                break
        memory["groups"][key] = {
            "samples": samples[:period],
            "phase": 0,
        }
    else:
        same = 0
        for j, old in enumerate(samples):
            if old == board:
                same = j
                break
        memory["groups"][key] = {"samples": samples, "phase": same}
    memory.pop("learning", None)


def _execute(result, memory, controls, available, pos):
    if result is None:
        return None
    action, key = result
    groups = memory["groups"]

    if action == "ACTION6" and key in controls and action in available:
        group = groups[key]
        group["phase"] = (group.get("phase", 0) + 1) % len(
            group["samples"]
        )
        r, c = controls[key]
        return action, int(r), int(c)

    if action in available:
        ids = sorted(groups)
        phases = tuple(
            groups[k].get("phase", 0) % len(groups[k]["samples"])
            for k in ids
        )
        memory["last_move"] = (pos, phases, action)
        return action, None, None
    return None


def act(obs, available, memory):
    obs = np.asarray(obs)

    if "layout" not in memory:
        memory["layout"] = _layout(obs)
    y0, panel = memory["layout"]
    board, rows, cols = _tiles(obs, y0, panel)
    controls = _controls(obs, panel)
    p = _player(obs)
    pos = None if p is None else _index(p[0], p[1], y0, rows, cols)

    if "crane_checked" not in memory:
        memory["crane_checked"] = True
        crane = _detect_crane(obs, y0, panel)
        if crane is not None:
            memory["crane"] = crane

    _update_observations(memory, board)
    groups = memory.setdefault("groups", {})

    last = memory.pop("last_move", None)
    if last is not None and pos == last[0]:
        memory.setdefault("forbidden", set()).add(last)

    crane = memory.get("crane")

    if crane is not None and crane["stage"] == "prepare":
        if not crane["attached"]:
            target = (crane["grab_x"], crane["grab_y"])
            if (crane["cx"], crane["cy"]) != target:
                click = _crane_action(crane, target=target)
            else:
                click = _crane_action(crane, grip=True)
            if click is not None and "ACTION6" in available:
                return "ACTION6", int(click[0]), int(click[1])
        else:
            target = (3, 0)
            if (crane["cx"], crane["cy"]) != target:
                click = _crane_action(crane, target=target)
                if click is not None and "ACTION6" in available:
                    return "ACTION6", int(click[0]), int(click[1])
            crane["stage"] = "middle"

    if crane is not None and crane["stage"] == "raise_right":
        target = (3, 3)
        if (crane["cx"], crane["cy"]) != target:
            click = _crane_action(crane, target=target)
            if click is not None and "ACTION6" in available:
                memory["forbidden"] = set()
                return "ACTION6", int(click[0]), int(click[1])
        crane["stage"] = "first_teleporter"

    if crane is not None and crane["stage"] == "teleport_pending":
        if pos != crane.get("teleport_from"):
            crane["stage"] = "second_side_prepare"
        else:
            crane["stage"] = "first_teleporter"

    if crane is not None and crane["stage"] == "second_side_prepare":
        target = (0, 0)
        if (crane["cx"], crane["cy"]) != target:
            click = _crane_action(crane, target=target)
            if click is not None and "ACTION6" in available:
                memory["forbidden"] = set()
                return "ACTION6", int(click[0]), int(click[1])
        crane["stage"] = "second_side"

    learning = memory.get("learning")
    if learning is not None and memory.pop("sampling_pending", False):
        key = learning["key"]
        ss = learning["samples"]

        if key == 2:
            if len(ss) >= 8:
                _finish_sampling(memory, key, ss, board, True)
                learning = None
            elif len(ss) == 1 and ss[0] == board:
                _finish_sampling(memory, key, ss, board, False)
                learning = None
            else:
                ss.append(board[:])
        else:
            repeated = any(old == board for old in ss)
            if repeated:
                _finish_sampling(memory, key, ss, board, False)
                learning = None
            elif len(ss) >= 8:
                ss.append(board[:])
                groups[key] = {"samples": ss, "phase": len(ss) - 1}
                memory.pop("learning", None)
                learning = None
            else:
                ss.append(board[:])

    if learning is not None:
        key = learning["key"]
        if key in controls and "ACTION6" in available:
            memory["sampling_pending"] = True
            r, c = controls[key]
            return "ACTION6", int(r), int(c)
        memory.pop("learning", None)

    unlearned = [k for k in sorted(controls) if k not in groups]
    if unlearned and "ACTION6" in available:
        current_history = memory.get("history", {}).get(pos)
        known_orange = (
            current_history is not None and 2 in _colours(current_history)
        )
        if not known_orange:
            move = _walk_to_orange(pos, board, rows, cols)
            if move in available:
                return move, None, None

        key = unlearned[0]
        memory["learning"] = {"key": key, "samples": [board[:]]}
        memory["sampling_pending"] = True
        r, c = controls[key]
        return "ACTION6", int(r), int(c)

    # The lower-right platform position connects the lower island to
    # a stable upper island. Stop there before raising the platform.
    if crane is not None and crane["stage"] == "middle":
        ox = crane["object_x"] + (3 - crane["grab_x"]) * 4
        oy = crane["object_y"] + crane["grab_y"] * 4
        middle = _index(ox + 6, oy - 2, y0, rows, cols)

        if pos == middle:
            crane["stage"] = "raise_right"
            target = (3, 3)
            click = _crane_action(crane, target=target)
            if click is not None and "ACTION6" in available:
                memory["forbidden"] = set()
                return "ACTION6", int(click[0]), int(click[1])
        elif middle is not None:
            result = _search(
                memory, board, rows, cols, pos, controls,
                extra_goals={middle}
            )
            executed = _execute(result, memory, controls, available, pos)
            if executed is not None:
                return executed

    if crane is not None and crane["stage"] == "first_teleporter":
        teleporters = memory.get("teleporters", set())
        if len(teleporters) >= 2:
            ox = crane["object_x"] + 4
            oy = crane["object_y"]
            first_tp = min(
                teleporters,
                key=lambda i: (
                    abs((i % cols) * 2 - ox)
                    + abs(y0 + (i // cols) * 2 - oy)
                )
            )

            if pos == first_tp and 6 in controls and 6 in groups:
                crane["stage"] = "teleport_pending"
                crane["teleport_from"] = pos
                executed = _execute(
                    ("ACTION6", 6), memory, controls, available, pos
                )
                if executed is not None:
                    return executed

            result = _search(
                memory, board, rows, cols, pos, controls,
                extra_goals={first_tp}
            )
            executed = _execute(result, memory, controls, available, pos)
            if executed is not None:
                return executed

    result = _search(memory, board, rows, cols, pos, controls)
    executed = _execute(result, memory, controls, available, pos)
    if executed is not None:
        return executed

    if pos is not None:
        ids = sorted(groups)
        phases = tuple(
            groups[k].get("phase", 0) % len(groups[k]["samples"])
            for k in ids
        )
        forbidden = memory.get("forbidden", set())
        visits = memory.setdefault("visits", {})
        visits[pos] = visits.get(pos, 0) + 1
        choices = []

        for nxt in _neighbours(pos, rows, cols):
            action = _direction(pos, nxt, cols)
            if action not in available or (pos, phases, action) in forbidden:
                continue
            safe, blocked = _status(board[nxt], dynamic=True)
            if safe and not blocked:
                choices.append((visits.get(nxt, 0), action))

        if choices:
            choices.sort()
            action = choices[0][1]
            memory["last_move"] = (pos, phases, action)
            return action, None, None

        for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if action in available and (pos, phases, action) not in forbidden:
                memory["last_move"] = (pos, phases, action)
                return action, None, None

    if "RESET" in available and not memory.get("reset_used"):
        memory.clear()
        memory["reset_used"] = True
        return "RESET", None, None

    for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        if action in available:
            return action, None, None
    return "RESET", None, None

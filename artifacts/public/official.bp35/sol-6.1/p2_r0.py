import math
import time
import heapq
from collections import deque
import numpy as np


def _read_frame(obs):
    ys, xs = np.where(obs[:63] == 9)
    if not len(xs):
        return None
    px = int(xs.min()) // 6
    top = int(ys.min()) - 1
    gravity = -1 if abs(top - 36) <= abs(top - 26) else 1
    cells = {}
    k0 = math.ceil((-top - 3) / 6)
    k1 = math.floor((61 - top) / 6)

    for k in range(k0, k1 + 1):
        oy = top + 6 * k
        for x in range(11):
            ox = 6 * x
            ya, yb = max(0, oy + 1), min(63, oy + 6)
            xa, xb = ox + 1, min(64, ox + 6)
            if yb <= ya or xb <= xa:
                continue
            a = obs[ya:yb, xa:xb]
            c = np.bincount(a.ravel(), minlength=16)
            area = a.size
            if area < 5:
                continue
            if x == px and k == 0:
                t = "p"
            elif c[14]:
                t = "x"
            elif c[8]:
                t = "g"
            elif c[12]:
                t = "b" if c[12] >= 0.45 * area else "t"
            elif c[7]:
                t = "*"
            elif c[15]:
                if c[11] or c[0]:
                    t = "s"
                else:
                    corners = 0
                    for dy, dx in ((1, 1), (1, 5), (5, 1), (5, 5)):
                        yy, xx = oy + dy, ox + dx
                        if 0 <= yy < 63 and 0 <= xx < 64:
                            corners += int(obs[yy, xx] == 3)
                    t = "y" if corners >= 2 and c[15] >= area * 0.55 else "m"
            elif c[10] > c[5]:
                t = "."
            elif c[5]:
                t = "#"
            else:
                t = "?"
            cells[(x, k)] = t
    return px, top, gravity, cells


def _view_key(board, px, py, gravity, segment):
    lo, hi = (-6, 3) if gravity == -1 else (-4, 5)
    return (
        segment, px, py, gravity,
        tuple(board.get((x, py + dy), "?")
              for dy in range(lo, hi + 1) for x in range(11))
    )


def _alignment(board, rel, boundary=None, direction=None):
    if not board:
        return None
    bys = [p[1] for p in board]
    rys = [p[1] for p in rel]
    low, high = min(bys) - max(rys), max(bys) - min(rys)
    best = None
    for offset in range(low, high + 1):
        if boundary is not None:
            if direction == -1 and offset > boundary:
                continue
            if direction == 1 and offset < boundary:
                continue
        good = bad = rare = 0
        score = 0.0
        patterns = {}
        for (x, dy), t in rel.items():
            if t in ("p", "?"):
                continue
            old = board.get((x, offset + dy))
            if old is None or old == "?":
                continue
            patterns.setdefault(dy, []).append((x, old))
            special = t not in (".", "#", "m") or old not in (".", "#", "m")
            weight = 5.0 if special else 1.0
            if t == old:
                good += 1
                score += weight
                rare += int(special)
            else:
                bad += 1
                score -= weight * 2.5
        diversity = len(set(tuple(v) for v in patterns.values()))
        if good < 25 or bad > max(2, good // 12):
            continue
        if rare == 0 and diversity < 3:
            continue
        if boundary is not None:
            score -= 0.035 * abs(offset - boundary)
        if best is None or score > best[0]:
            best = score, offset
    return best


def _bit_indices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


class _Model:
    def __init__(self, board, px, py, gravity):
        self.board = board
        ys = [p[1] for p in board]
        self.ymin = min(ys + [py]) - 2
        self.ymax = max(ys + [py]) + 2
        self.size = (self.ymax - self.ymin + 1) * 11
        self.base = ["?"] * self.size
        self.bits = [1 << i for i in range(self.size)]
        self.initial_blue = 0
        self.initial_yellow = 0
        self.specials = []
        self.goals = []
        self.greens = 0
        for (x, y), t in board.items():
            if not 0 <= x < 11:
                continue
            i = self.index(x, y)
            if t == "p":
                t = "."
            self.base[i] = t
            if t == "b":
                self.initial_blue |= self.bits[i]
            elif t == "y":
                self.initial_yellow |= self.bits[i]
            elif t == "g":
                self.greens |= self.bits[i]
            if t in ("g", "b", "t"):
                self.specials.append(i)
            if t == "*":
                self.goals.append(i)
        self.initial = (
            self.index(px, py), gravity, 0,
            self.initial_blue, self.initial_yellow
        )

    def index(self, x, y):
        return (y - self.ymin) * 11 + x

    def xy(self, i):
        return i % 11, i // 11 + self.ymin

    def tile(self, i, state):
        if not 0 <= i < self.size:
            return "?"
        bit = self.bits[i]
        if state[4] & bit:
            return "y"
        t = self.base[i]
        if t in ("x", "g"):
            return "." if state[2] & bit else t
        if t in ("b", "t"):
            return "b" if state[3] & bit else "t"
        if t == "y":
            return "."
        return t

    def visible(self, i, state):
        x, y = self.xy(i)
        _, py = self.xy(state[0])
        anchor = 36 if state[1] == -1 else 26
        sy = anchor + (y - py) * 6 + 3
        return 0 <= x <= 10 and 0 <= sy <= 62

    def fall(self, start, state):
        p = start
        step = 11 * state[1]
        for _ in range(self.size + 2):
            nxt = p + step
            t = self.tile(nxt, state)
            if t in (".", "t", "m"):
                p = nxt
            elif t == "*":
                return nxt, 2
            elif t == "s":
                return p, -1
            elif t == "?":
                return nxt, 1
            else:
                return p, 0
        return p, 1

    def transition(self, state, action):
        p, gravity, removed, bridges, yellow = state
        kind, target = action
        if kind in (3, 4):
            x, _ = self.xy(p)
            dx = -1 if kind == 3 else 1
            if not 0 <= x + dx < 11:
                return None
            dest = p + dx
            t = self.tile(dest, state)
            if t == "*":
                return (dest, gravity, removed, bridges, yellow), 2
            if t == "?":
                return (dest, gravity, removed, bridges, yellow), 1
            if t not in (".", "t", "m"):
                return None
            dest, outcome = self.fall(dest, state)
            return (dest, gravity, removed, bridges, yellow), outcome

        if kind != 6 or not self.visible(target, state):
            return None
        t = self.tile(target, state)
        if t not in ("x", "g", "b", "t", "y"):
            return None
        bit = self.bits[target]
        outcome = 0

        if t == "g":
            gravity = -gravity
            falling_state = (p, gravity, removed, bridges, yellow)
            support = p + 11 * gravity
            st = self.tile(support, falling_state)
            if st == "?":
                p, outcome = support, 1
            elif st not in ("#", "x", "g", "b", "y"):
                p, outcome = self.fall(support, falling_state)
            removed |= bit

        elif t == "y":
            tx, _ = self.xy(target)
            for delta in (-11, 11, -1, 1):
                if delta == -1 and tx == 0:
                    continue
                if delta == 1 and tx == 10:
                    continue
                n = target + delta
                if (
                    0 <= n < self.size and n != p
                    and self.tile(n, state) == "."
                ):
                    yellow |= self.bits[n]
            growing = (p, gravity, removed, bridges, yellow)
            if target == p + 11 * gravity:
                p, outcome = self.fall(target, growing)
            yellow &= ~bit

        else:
            if target == p + 11 * gravity:
                p, outcome = self.fall(target, state)
            if t == "x":
                removed |= bit
            else:
                bridges ^= bit
        return (p, gravity, removed, bridges, yellow), outcome

    def actions(self, state, allowed, local_growth=False):
        p, gravity, _, _, yellow = state
        px, py = self.xy(p)
        actions = []
        if 3 in allowed:
            actions.append((3, -1))
        if 4 in allowed:
            actions.append((4, -1))
        if 6 not in allowed:
            return actions
        targets = set()
        for i in (p + 11 * gravity, p - 1, p + 1):
            if 0 <= i < self.size:
                x, _ = self.xy(i)
                if abs(x - px) <= 1 and self.tile(i, state) in ("x", "b"):
                    targets.add(i)

        visible_green = any(
            self.tile(i, state) == "g" and self.visible(i, state)
            for i in self.specials
        )
        for i in self.specials:
            if not self.visible(i, state):
                continue
            t = self.tile(i, state)
            x, y = self.xy(i)
            if t == "g":
                targets.add(i)
            elif t == "t" and abs(x - px) <= 1:
                dy = y - py
                if not dy or (dy * gravity < 0 and not visible_green):
                    continue
                direction = 1 if dy > 0 else -1
                clear = True
                for yy in range(py, y, direction):
                    q = self.index(x, yy)
                    if q != p and self.tile(q, state) not in (".", "t", "m"):
                        clear = False
                        break
                if clear:
                    targets.add(i)

        for i in _bit_indices(yellow):
            if not self.visible(i, state):
                continue
            x, y = self.xy(i)
            if not local_growth or (abs(x - px) <= 3 and abs(y - py) <= 4):
                targets.add(i)
        actions.extend((6, i) for i in sorted(targets))
        return actions

    def passed_goal(self, state):
        if not self.goals or self.greens & ~state[2]:
            return False
        _, py = self.xy(state[0])
        return all(
            (self.xy(i)[1] - py) * state[1] < 0 for i in self.goals
        )

    def information(self, state):
        _, py = self.xy(state[0])
        lo, hi = (-6, 3) if state[1] == -1 else (-4, 5)
        unknown = 0
        for y in range(py + lo, py + hi + 1):
            for x in range(1, 10):
                i = self.index(x, y)
                if not 0 <= i < self.size or self.base[i] == "?":
                    unknown += 1
        return unknown

    def changed_board(self, state):
        result = self.board.copy()
        for i, old in enumerate(self.base):
            if old != "?":
                result[self.xy(i)] = self.tile(i, state)
        return result


def _path(nodes, n):
    result = []
    while n:
        state, parent, action, depth, flips = nodes[n]
        result.append(action)
        n = parent
    result.reverse()
    return result


def _external_internal(model, action):
    if action[0] == 6:
        x, y = model.xy(action[1])
        return 6, x, y
    return action[0], None, None


def _goal_search(model, allowed, bans, deadline):
    if not model.goals:
        return []
    initial = model.initial

    def heuristic(s):
        x, y = model.xy(s[0])
        remaining = bool(model.greens & ~s[2])
        best = float("inf")
        for i in model.goals:
            gx, gy = model.xy(i)
            if (gy - y) * s[1] < 0 and not remaining:
                continue
            h = abs(x - gx) + 0.45 * abs(y - gy)
            if (gy - y) * s[1] < 0:
                h += 8
            best = min(best, h)
        return best

    h = heuristic(initial)
    if not math.isfinite(h):
        return []
    nodes = [(initial, -1, None, 0, 0)]
    heap = [(h, 0)]
    costs = {initial: 0}
    expanded = 0

    while heap and expanded < 42000 and len(nodes) < 110000:
        if expanded % 128 == 0 and time.perf_counter() >= deadline:
            break
        _, n = heapq.heappop(heap)
        state, _, _, depth, _ = nodes[n]
        if costs.get(state) != depth:
            continue
        expanded += 1
        if depth >= 65:
            continue

        for action in model.actions(state, allowed, True):
            if n == 0 and _external_internal(model, action) in bans:
                continue
            tr = model.transition(state, action)
            if tr is None:
                continue
            ns, outcome = tr
            if outcome in (-1, 1):
                continue
            nd = depth + 1
            if costs.get(ns, 100000) <= nd:
                continue
            hh = heuristic(ns)
            if not math.isfinite(hh) and outcome != 2:
                continue
            costs[ns] = nd
            ni = len(nodes)
            nodes.append((ns, n, action, nd, 0))
            if outcome == 2:
                return [
                    _external_internal(model, a)
                    for a in _path(nodes, ni)
                ]
            heapq.heappush(heap, (nd + 1.7 * hh, ni))
    return []


def _search(model, allowed, memory, root_key, segment, deadline):
    initial = model.initial
    bans = {tuple(a) for a in memory.get("bans", {}).get(root_key, ())}
    if model.goals:
        goal_deadline = min(deadline, time.perf_counter() + 3.8)
        plan = _goal_search(model, allowed, bans, goal_deadline)
        if plan:
            return plan

    nodes = [(initial, -1, None, 0, 0)]
    queue = deque([0])
    seen = {initial}
    visits = memory.get("visits", {})
    safe = blind = fallback = None
    first_frontier = None
    _, root_y = model.xy(initial[0])
    root_g = initial[1]
    controlled = bool(model.initial_yellow)
    expanded = 0
    truncated = False
    limit = 32000 if controlled else 18000

    def path(n):
        return [_external_internal(model, a) for a in _path(nodes, n)]

    while queue and expanded < limit and len(nodes) < 80000:
        if expanded % 128 == 0 and time.perf_counter() >= deadline:
            truncated = True
            break
        n = queue.popleft()
        state, _, _, depth, flips = nodes[n]
        expanded += 1
        if depth >= 34:
            continue
        if first_frontier is not None:
            extra = 12 if model.goals else 5
            if depth > first_frontier + extra:
                break

        for action in model.actions(state, allowed, controlled):
            if n == 0 and _external_internal(model, action) in bans:
                continue
            tr = model.transition(state, action)
            if tr is None:
                continue
            ns, outcome = tr
            if outcome == -1 or ns in seen:
                continue
            # Do not consume the final reversal and overshoot a known goal.
            if controlled and outcome != 2 and model.passed_goal(ns):
                continue
            seen.add(ns)
            nd = depth + 1
            nf = flips + int(ns[1] != state[1])
            ni = len(nodes)
            nodes.append((ns, n, action, nd, nf))
            if outcome == 2:
                return path(ni)

            x, y = model.xy(ns[0])
            penalty = 0.65 * min(6, visits.get((segment, x, y, ns[1]), 0))
            progress = (y - root_y) * root_g
            info = model.information(ns)

            if outcome == 1:
                value = (
                    8 + 0.5 * min(6, max(0, progress))
                    - 1.7 * nd - 2 * nf - penalty
                )
                if blind is None or value > blind[0]:
                    blind = value, ni
                if not controlled and first_frontier is None:
                    first_frontier = nd
                continue

            queue.append(ni)
            if info >= 6:
                if first_frontier is None:
                    first_frontier = nd
                value = (
                    10 + 0.13 * min(info, 90)
                    + 0.5 * min(6, max(0, progress))
                    - 1.7 * nd - 2 * nf - penalty
                )
                if safe is None or value > safe[0]:
                    safe = value, ni

            value = 3 * progress - 0.8 * nd - penalty - 2 * nf
            if progress > 0 and (fallback is None or value > fallback[0]):
                fallback = value, ni

    truncated |= expanded >= limit or len(nodes) >= 80000
    if controlled:
        if safe is not None:
            return path(safe[1])
        if truncated and fallback is not None:
            return path(fallback[1])
        return []
    choices = [c for c in (safe, blind) if c is not None]
    if choices:
        return path(max(choices)[1])
    if truncated and fallback is not None:
        return path(fallback[1])
    return []


def _output(action, py, top, available):
    kind, x, y = action
    name = "ACTION" + str(kind)
    if name not in available:
        return None
    if kind == 6:
        row, col = top + 6 * (y - py) + 3, 6 * x + 3
        if not (0 <= row <= 62 and 0 <= col <= 63):
            return None
        return name, int(row), int(col)
    return name, None, None


def act(obs, available, memory):
    deadline = time.perf_counter() + 8.0
    available = {str(a).upper() for a in available}
    allowed = {
        int(a[6:]) for a in available
        if a.startswith("ACTION") and a[6:].isdigit()
    }
    reading = _read_frame(obs)
    if reading is None:
        if "RESET" in available:
            memory.clear()
            return "RESET", None, None
        return next(iter(sorted(available)), "ACTION3"), None, None

    px, top, gravity, relative = reading
    board = memory.get("board", {})
    segment = memory.get("segment", 0)
    prediction = memory.pop("prediction", None)
    aligned = True

    if not board:
        py = 0
    elif prediction is not None and not prediction.get("unknown", False):
        py = prediction["y"]
        if prediction.get("x") != px or prediction["gravity"] != gravity:
            memory["plan"] = []
    elif prediction is not None:
        candidate = _alignment(
            board, relative, prediction["y"], prediction["gravity"]
        )
        chosen_board, chosen_segment = board, segment
        if candidate is None:
            best = None
            for old_segment, old_board in memory.get("archives", []):
                c = _alignment(old_board, relative)
                if c is not None and (best is None or c[0] > best[0]):
                    best = c
                    chosen_board, chosen_segment = old_board, old_segment
            candidate = best
        if candidate is not None:
            if chosen_segment != segment:
                archives = memory.setdefault("archives", [])
                archives.append((segment, board.copy()))
                memory["archives"] = archives[-10:]
            py = candidate[1]
            board, segment = chosen_board.copy(), chosen_segment
        else:
            archives = memory.setdefault("archives", [])
            archives.append((segment, board.copy()))
            memory["archives"] = archives[-10:]
            segment = memory.get("next_segment", segment + 1)
            memory["next_segment"] = segment + 1
            board, py, aligned = {}, 0, False
        memory["plan"] = []
    else:
        py = memory.get("player_y", 0)

    old_goals = sum(t == "*" for t in board.values())
    discrepancies = 0
    for (x, dy), t in relative.items():
        key = x, py + dy
        if t == "?":
            continue
        if t == "p":
            if board.get(key) not in ("t", "m"):
                board[key] = "."
        else:
            old = board.get(key)
            if old is not None and old != t:
                discrepancies += 1
            board[key] = t

    if discrepancies >= 3 or sum(t == "*" for t in board.values()) > old_goals:
        memory["plan"] = []

    memory.update({
        "board": board, "segment": segment,
        "player_y": py, "player_x": px, "gravity": gravity
    })
    root_key = _view_key(board, px, py, gravity, segment)
    visits = memory.setdefault("visits", {})
    position_key = segment, px, py, gravity
    visits[position_key] = visits.get(position_key, 0) + 1

    model = _Model(board, px, py, gravity)
    plan = memory.get("plan", [])
    if plan:
        first = tuple(plan[0])
        target = model.index(first[1], first[2]) if first[0] == 6 else -1
        tr = model.transition(model.initial, (first[0], target))
        out = _output(first, py, top, available)
        if tr is None or tr[1] == -1 or out is None or not aligned:
            plan = []

    if not plan:
        plan = _search(model, allowed, memory, root_key, segment, deadline)

    if not plan:
        history = memory.setdefault("history", [])
        if history and "ACTION7" in available:
            previous = history.pop()
            bans = memory.setdefault("bans", {})
            entry = bans.setdefault(previous["key"], [])
            if previous["action"] not in entry:
                entry.append(previous["action"])
            if previous["segment"] == segment:
                restored = board.copy()
                restored.update(previous["board"])
            else:
                archives = memory.setdefault("archives", [])
                archives.append((segment, board.copy()))
                memory["archives"] = archives[-10:]
                restored = previous["board"].copy()
            memory.update({
                "board": restored,
                "segment": previous["segment"],
                "player_y": previous["y"],
                "player_x": previous["x"],
                "gravity": previous["gravity"],
                "prediction": {
                    "x": previous["x"], "y": previous["y"],
                    "gravity": previous["gravity"], "unknown": False
                },
                "plan": []
            })
            return "ACTION7", None, None

        if "RESET" in available:
            memory.clear()
            return "RESET", None, None
        for kind in (4, 3):
            if kind in allowed:
                return "ACTION" + str(kind), None, None
        return next(iter(sorted(available)), "ACTION3"), None, None

    action = tuple(plan[0])
    target = model.index(action[1], action[2]) if action[0] == 6 else -1
    transition = model.transition(model.initial, (action[0], target))
    output = _output(action, py, top, available)
    if transition is None or output is None:
        memory["plan"] = []
        return next(
            (a for a in ("ACTION3", "ACTION4", "RESET") if a in available),
            next(iter(sorted(available)), "RESET")
        ), None, None

    ns, outcome = transition
    nx, ny = model.xy(ns[0])
    memory.setdefault("history", []).append({
        "board": board.copy(), "segment": segment,
        "x": px, "y": py, "gravity": gravity,
        "key": root_key, "action": action
    })
    memory["board"] = model.changed_board(ns)
    memory["prediction"] = {
        "x": nx, "y": ny, "gravity": ns[1], "unknown": outcome == 1
    }
    memory["plan"] = list(plan[1:]) if outcome == 0 else []
    return output

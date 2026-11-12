import math
import time
import heapq
from collections import deque
import numpy as np

EMPTY, WALL, BREAK, REVERSE, SOLID, GHOST, GROW, SPIKE, GOAL, WATER = range(10)
UNKNOWN = -1
PASSABLE = (EMPTY, GHOST, WATER)


def _templates():
    out = []

    def add(kind, rows, palette):
        a = np.full((7, 7), 10, dtype=np.int16)
        for y, row in enumerate(rows):
            for x, c in enumerate(row):
                if c in palette:
                    a[y, x] = palette[c]
        out.append((kind, a[1:6, 1:6]))

    add(EMPTY, [""] * 7, {})
    add(WALL, [
        "xxxxxxx", "xxoxxxx", "xxxxxxx", "xxxxxxx",
        "xoxxxxx", "xxxxoxx", "xxxxxxx"
    ], {"x": 5, "o": 3})

    block = [
        ".......", ".oxxxo.", ".xxxxx.", ".xxxxx.",
        ".xxxxx.", ".oxxxo.", "......."
    ]
    for kind, color in ((BREAK, 14), (REVERSE, 8),
                        (SOLID, 12), (GROW, 15)):
        add(kind, block, {".": 5, "o": 3, "x": color})

    add(GHOST, [
        ".......", "..xxx..", ".xoxox.", ".xxoxx.",
        ".xoxox.", "..xxx..", "......."
    ], {"o": 12})

    spike = [
        " ooooo", " oxrxo", " o...o", " o...o",
        " o...o", " o...o", "       "
    ]
    for rows in (spike, spike[::-1]):
        add(SPIKE, rows, {"x": 11, "o": 5, ".": 15, "r": 0})

    add(GOAL, [
        "  ... ", " ..x..", " .xxx.", " ..x..",
        "  ... ", "      ", "      "
    ], {"x": 7, ".": 5})

    add(WATER, [
        ".ooooo", "..oo.o", "......", " .... ",
        "      ", "      ", "      "
    ][::-1], {"o": 15, ".": 5})

    add(WALL, [
        "oooo.o", "o.oooo", "o.o.oo", "o.o.oo",
        "o.oooo", "oooo.o", "oooo.o"
    ][::-1], {"o": 15, ".": 5})
    return out


_TEMPLATES = _templates()


def _decode(obs):
    pp = np.argwhere(obs[:63] == 9)
    if len(pp) == 0:
        return None
    px = int(np.median(pp[:, 1])) // 6
    top = int(pp[:, 0].min()) - 1
    gravity = -1 if abs(top - 36) <= abs(top - 26) else 1
    cells = {}

    for ry in range(math.floor(-top / 6) - 1,
                    math.ceil((63 - top) / 6) + 2):
        sy = top + 6 * ry
        rr = [r for r in range(1, 6) if 0 <= sy + r < 63]
        if not rr:
            continue
        for x in range(11):
            cc = [c for c in range(1, 6) if x * 6 + c < 64]
            if not cc:
                continue
            a = obs[np.ix_([sy + r for r in rr],
                           [x * 6 + c for c in cc])]
            if x == px and ry == 0:
                blue = int(np.count_nonzero(a == 12))
                cells[ry * 16 + x] = (
                    SOLID if blue > 8 else GHOST if blue else EMPTY
                )
                continue
            best = 1000
            kinds = set()
            for kind, template in _TEMPLATES:
                b = template[np.ix_([r - 1 for r in rr],
                                    [c - 1 for c in cc])]
                error = int(np.count_nonzero(a != b))
                if error < best:
                    best, kinds = error, {kind}
                elif error == best:
                    kinds.add(kind)
            if best <= max(2, int(a.size * .28)) and len(kinds) == 1:
                cells[ry * 16 + x] = next(iter(kinds))
    return px, top, gravity, cells


def _visible(position, gravity, target):
    py = position // 16
    ty, tx = divmod(target, 16)
    sy = (36 if gravity == -1 else 26) + 6 * (ty - py) + 3
    return 0 <= tx <= 10 and 0 <= sy <= 63


def _signature(world, position, gravity):
    py = position // 16
    return position, gravity, tuple(
        world.get(y * 16 + x, UNKNOWN)
        for y in range(py - 5, py + 6)
        for x in range(1, 10)
    )


def _register(memory, px, gravity, relative):
    world = memory.get("world")
    if world is None:
        return 0, {}, True

    previous = memory.get("expected")
    if previous is None:
        return memory.get("py", 0), world, False

    pred_y, pred_x = divmod(previous["position"], 16)
    old_y = memory.get("py", 0)
    mismatch = pred_x != px or previous["gravity"] != gravity

    if previous["flag"] != 1 and not mismatch:
        return pred_y, world, False

    if mismatch:
        candidates = range(old_y - 50, old_y + 51)
    else:
        candidates = [pred_y + gravity * d for d in range(70)]

    def weight(t):
        return .35 if t in (EMPTY, WALL, WATER) else 2.0

    best_y, best_score = pred_y, -1e100
    for y in candidates:
        score = -.035 * abs(y - pred_y)
        for rp, tile in relative.items():
            ry, x = divmod(rp, 16)
            if ry == 0 and x == px:
                continue
            old = world.get((y + ry) * 16 + x, UNKNOWN)
            if old == UNKNOWN:
                continue
            if old == tile:
                score += weight(tile)
            else:
                score -= 2.5 * max(weight(old), weight(tile))
        if score > best_score:
            best_y, best_score = y, score
    return best_y, world, True


class _Model:
    def __init__(self, world, allowed):
        self.world = world
        self.allowed = allowed
        self.special = {
            p for p, t in world.items() if t in (REVERSE, GHOST, GROW)
        }
        self.grows = {p for p, t in world.items() if t == GROW}
        self.reds = {p for p, t in world.items() if t == REVERSE}
        self.goals = [p for p, t in world.items() if t == GOAL]
        self.distance = self._goal_distances()

    def tile(self, p, changes):
        if not 0 <= p % 16 <= 10:
            return WALL
        return changes.get(p, self.world.get(p, UNKNOWN))

    def put(self, changes, p, tile):
        if self.world.get(p, UNKNOWN) == tile:
            changes.pop(p, None)
        else:
            changes[p] = tile

    def _goal_distances(self):
        if not self.goals:
            return {}
        ys = [p // 16 for p in self.world]
        low, high = min(ys) - 2, max(ys) + 2
        dist = {p: 0.0 for p in self.goals}
        heap = [(0.0, p) for p in self.goals]
        heapq.heapify(heap)
        while heap:
            d, p = heapq.heappop(heap)
            if d != dist[p]:
                continue
            for delta in (-1, 1, -16, 16):
                q = p + delta
                y, x = divmod(q, 16)
                if not (0 <= x <= 10 and low <= y <= high):
                    continue
                t = self.world.get(q, UNKNOWN)
                if t in (WALL, SPIKE):
                    continue
                nd = d + (1.0 if abs(delta) == 1 else .55)
                if t in (BREAK, SOLID, GROW):
                    nd += .25
                if nd < dist.get(q, 1e100):
                    dist[q] = nd
                    heapq.heappush(heap, (nd, q))
        return dist

    def heuristic(self, state):
        if not self.goals:
            return 0.0
        p, gravity, packed = state
        py, px = divmod(p, 16)
        fallback = min(abs(g % 16 - px) + .55 * abs(g // 16 - py)
                       for g in self.goals)
        h = self.distance.get(p, fallback)
        if all((g // 16 - py) * gravity < 0 for g in self.goals):
            changes = dict(packed)
            reds_left = any(self.tile(r, changes) == REVERSE
                            for r in self.reds)
            h += 2.0 if reds_left else 11.0
        return h

    def fall(self, start, gravity, changes):
        p = start
        step = gravity * 16
        for _ in range(110):
            q = p + step
            tile = self.tile(q, changes)
            if tile == UNKNOWN:
                return p, 1
            if tile in PASSABLE:
                p = q
            elif tile == GOAL:
                return q, 2
            elif tile == SPIKE:
                return p, -1
            else:
                return p, 0
        return p, 1

    def transition(self, state, action):
        p, gravity, packed = state
        changes = dict(packed)
        kind, target = action

        if kind in (3, 4):
            q = p + (-1 if kind == 3 else 1)
            tile = self.tile(q, changes)
            if tile == GOAL:
                return (q, gravity, packed), 2
            if tile not in PASSABLE:
                return None
            landing, flag = self.fall(q, gravity, changes)
            if flag < 0:
                return None
            return (landing, gravity, packed), flag

        if kind != 6 or target == p:
            return None
        if not _visible(p, gravity, target):
            return None
        tile = self.tile(target, changes)
        if tile not in (BREAK, REVERSE, SOLID, GHOST, GROW):
            return None

        if tile == GROW:
            for q in (target - 1, target + 1, target - 16, target + 16):
                if q != p and self.tile(q, changes) == EMPTY:
                    self.put(changes, q, GROW)

        landing, flag = p, 0
        if tile == REVERSE:
            gravity = -gravity
            support = p + gravity * 16
            t = self.tile(support, changes)
            if t not in (WALL, REVERSE, SOLID, BREAK, GROW):
                if t == UNKNOWN:
                    flag = 1
                else:
                    landing, flag = self.fall(support, gravity, changes)
        elif target == p + gravity * 16:
            landing, flag = self.fall(target, gravity, changes)

        if flag < 0:
            return None
        replacement = (
            GHOST if tile == SOLID else SOLID if tile == GHOST else EMPTY
        )
        self.put(changes, target, replacement)
        return (landing, gravity, tuple(sorted(changes.items()))), flag

    def actions(self, state):
        p, gravity, packed = state
        changes = dict(packed)
        actions = []
        if 3 in self.allowed:
            actions.append((3, 0))
        if 4 in self.allowed:
            actions.append((4, 0))
        if 6 not in self.allowed:
            return actions

        clicks = set()
        for q in (p - 1, p + 1, p + gravity * 16):
            if self.tile(q, changes) in (BREAK, SOLID, GHOST, GROW):
                clicks.add(q)

        special = self.special | {
            q for q, t in changes.items() if t in (REVERSE, GHOST)
        }
        py = p // 16
        for q in special:
            if q == p or not _visible(p, gravity, q):
                continue
            tile = self.tile(q, changes)
            if tile == REVERSE:
                clicks.add(q)
            elif tile == GHOST:
                y, x = divmod(q, 16)
                distance = (y - py) * gravity
                if distance <= 0:
                    continue
                clear = True
                for d in range(distance):
                    r = (py + gravity * d) * 16 + x
                    if self.tile(r, changes) not in PASSABLE:
                        clear = False
                        break
                if clear:
                    clicks.add(q)

        actions.extend((6, q) for q in sorted(clicks)
                       if q != p and _visible(p, gravity, q))
        return actions

    def growth_edges(self, state):
        """Build a selected landing surface, then move/fall onto it."""
        if 6 not in self.allowed or not self.grows:
            return []
        p, gravity, packed = state
        changes = dict(packed)

        sources = self.grows | {
            q for q, t in changes.items() if t == GROW
        }
        sources = sorted(q for q in sources
                         if q != p and q != p + gravity * 16
                         and self.tile(q, changes) == GROW
                         and _visible(p, gravity, q))
        if not sources:
            return []

        # Multi-source shortest growth paths. A path's final tile is left
        # intact as the desired support; preceding tiles are clicked.
        parent = {q: None for q in sources}
        depth = {q: 0 for q in sources}
        queue = deque(sources)
        while queue:
            q = queue.popleft()
            if depth[q] >= 6:
                continue
            for r in (q - 1, q + 1, q - 16, q + 16):
                if r in parent or r == p:
                    continue
                if not _visible(p, gravity, r):
                    continue
                if self.tile(r, changes) != EMPTY:
                    continue
                parent[r] = q
                depth[r] = depth[q] + 1
                queue.append(r)

        targets = []
        # Horizontal movement that would otherwise fall past a useful height.
        for kind, delta in ((3, -1), (4, 1)):
            if kind not in self.allowed:
                continue
            start = p + delta
            if self.tile(start, changes) not in PASSABLE:
                continue
            for d in range(1, 8):
                q = start + gravity * 16 * d
                t = self.tile(q, changes)
                if t not in PASSABLE:
                    break
                if t == EMPTY and q in parent and depth[q] > 0:
                    targets.append((q, (kind, 0)))

        support = p + gravity * 16
        if self.tile(support, changes) in (BREAK, SOLID, GHOST, GROW):
            for d in range(2, 8):
                q = p + gravity * 16 * d
                t = self.tile(q, changes)
                if t not in PASSABLE:
                    break
                if t == EMPTY and q in parent and depth[q] > 0:
                    targets.append((q, (6, support)))

        # A stopper can also make a gravity reversal safe and useful.
        reds = [q for q in self.reds
                if self.tile(q, changes) == REVERSE
                and _visible(p, gravity, q)]
        if reds:
            for d in range(1, 8):
                q = p - gravity * 16 * d
                t = self.tile(q, changes)
                if t not in PASSABLE:
                    break
                if t == EMPTY and q in parent and depth[q] > 0:
                    for red in reds:
                        targets.append((q, (6, red)))

        results = []
        seen = set()
        for target, trigger in targets:
            route = []
            q = parent[target]
            while q is not None:
                route.append((6, q))
                q = parent[q]
            route.reverse()
            route.append(trigger)

            current = state
            flag = 0
            valid = True
            for action in route:
                result = self.transition(current, action)
                if result is None:
                    valid = False
                    break
                current, flag = result
                if flag != 0:
                    valid = flag == 2
                    break
            if not valid:
                continue
            if current[:2] == state[:2] and flag != 2:
                continue
            if current in seen:
                continue
            seen.add(current)
            results.append((tuple(route), current, flag))
        return results

    def edges(self, state):
        result = []
        for a in self.actions(state):
            transition = self.transition(state, a)
            if transition is not None:
                nxt, flag = transition
                result.append(((a,), nxt, flag))
        result.extend(self.growth_edges(state))
        return result


def _search(model, root, banned, repeats, deadline):
    world = model.world
    start_y = root[0] // 16
    unknown_cache = {}

    def unknown_view(p, gravity):
        y = p // 16
        key = y, gravity
        if key not in unknown_cache:
            offsets = range(-6, 5) if gravity == -1 else range(-4, 6)
            unknown_cache[key] = sum(
                world.get((y + dy) * 16 + x, UNKNOWN) == UNKNOWN
                for dy in offsets for x in range(1, 10)
            )
        return unknown_cache[key]

    initial_unknown = unknown_view(root[0], root[1])
    records = [(root, -1, (), 0.0)]
    costs = {root: 0.0}
    heap = [(1.6 * model.heuristic(root), 0, 0)]
    counter = 0

    safe = None
    safe_score = float("inf")
    risky = None
    risky_score = float("inf")
    cost_limit = float("inf")
    progress = None
    progress_score = model.heuristic(root)
    expanded = 0

    def path(index, tail=()):
        chunks = [tail] if tail else []
        while index > 0:
            state, parent, actions, cost = records[index]
            chunks.append(actions)
            index = parent
        answer = []
        for chunk in reversed(chunks):
            answer.extend(chunk)
        return answer

    while heap:
        _, _, index = heapq.heappop(heap)
        state, parent, incoming, cost = records[index]
        if costs.get(state) != cost or cost > cost_limit:
            continue
        expanded += 1
        if expanded % 16 == 0:
            if (time.perf_counter() >= deadline
                    or expanded > 22000 or len(records) > 100000):
                break

        for actions, nxt, flag in model.edges(state):
            if index == 0 and actions[0] in banned:
                continue
            penalty = 0.0
            if index == 0:
                penalty = 4.0 * repeats.get(actions[0], 0)
            nc = cost + len(actions) + .015 * sum(
                a[0] == 6 for a in actions
            ) + penalty

            if flag == 2:
                return path(index, actions)

            gain = unknown_view(nxt[0], nxt[1]) - initial_unknown
            vertical = nxt[0] // 16 != start_y
            h = model.heuristic(nxt)

            if flag == 1:
                score = (
                    nc + .10 * h
                    - .35 * min(max(gain, 0) / 9.0, 5.0)
                )
                if score < risky_score:
                    risky_score = score
                    risky = index, actions
                continue

            if vertical and gain >= 6:
                score = (
                    nc + .10 * h
                    - .70 * min(max(gain, 0) / 9.0, 5.0)
                )
                if score < safe_score:
                    safe_score = score
                    safe = index, actions
                cost_limit = min(cost_limit, nc + 5.0)

            if nc >= costs.get(nxt, float("inf")):
                continue
            costs[nxt] = nc
            ni = len(records)
            records.append((nxt, index, actions, nc))

            score = h + .12 * nc
            if model.goals and score < progress_score - .4:
                progress_score = score
                progress = ni

            counter += 1
            heapq.heappush(heap, (nc + 1.6 * h, counter, ni))

    if safe is not None:
        return path(*safe)

    # Continue exploration rather than oscillating toward a goal which
    # requires a switch in a yet-unseen section of the map.
    if risky is not None:
        return path(*risky)

    if progress is not None:
        return path(progress)
    return None


def act(obs, available, memory):
    obs = np.asarray(obs)
    available_set = set(available)

    def reset_or_wait():
        memory.clear()
        if "RESET" in available_set:
            return "RESET", None, None
        if "ACTION7" in available_set:
            return "ACTION7", None, None
        name = next((a for a in available if a != "ACTION6"), available[0])
        return (name, 32, 32) if name == "ACTION6" else (name, None, None)

    decoded = _decode(obs)
    if decoded is None:
        return reset_or_wait()

    px, top, gravity, relative = decoded
    py, world, changed_registration = _register(
        memory, px, gravity, relative
    )
    position = py * 16 + px
    if changed_registration:
        memory.pop("plan", None)

    meaningful_contradictions = 0
    for rp, tile in relative.items():
        ry, x = divmod(rp, 16)
        p = (py + ry) * 16 + x
        if ry == 0 and x == px:
            if p not in world:
                world[p] = tile
            continue
        old = world.get(p, UNKNOWN)
        if old != UNKNOWN and old != tile:
            if old not in (EMPTY, WALL, WATER) or tile not in (
                    EMPTY, WALL, WATER):
                meaningful_contradictions += 1
        world[p] = tile

    if meaningful_contradictions:
        memory.pop("plan", None)

    memory["world"] = world
    memory["py"] = py

    allowed = {n for n in (3, 4, 6)
               if "ACTION%d" % n in available_set}
    model = _Model(world, allowed)
    root = position, gravity, ()
    signature = _signature(world, position, gravity)

    bad = memory.setdefault("bad", {})
    banned = set(tuple(a) for a in bad.get(signature, ()))
    attempts = memory.setdefault("attempts", {})
    repeats = attempts.get(signature, {})

    last = memory.get("last")
    previous_frame = memory.get("previous_frame")
    if (last is not None and previous_frame is not None
            and last[0] != 7
            and np.array_equal(obs[:63], previous_frame)):
        banned.add(tuple(last))
        bad[signature] = list(banned)
        memory.pop("plan", None)

    for action, count in repeats.items():
        if count >= 3:
            banned.add(action)

    plan = memory.get("plan", [])
    if plan:
        first = tuple(plan[0])
        if (first in banned or first[0] not in allowed
                or model.transition(root, first) is None):
            plan = []

    if not plan:
        plan = _search(model, root, banned, repeats,
                       time.perf_counter() + 5.2) or []

    history = memory.setdefault("history", [])

    if not plan:
        if history and "ACTION7" in available_set:
            old = history.pop()
            for p, tile in old["before"]:
                if tile == UNKNOWN:
                    world.pop(p, None)
                else:
                    world[p] = tile

            restored = old["position"]
            restored_g = old["gravity"]
            sig = _signature(world, restored, restored_g)
            entries = set(tuple(a) for a in bad.get(sig, ()))
            entries.add(tuple(old["action"]))
            bad[sig] = list(entries)
            memory["expected"] = {
                "position": restored, "gravity": restored_g, "flag": 0
            }
            memory["plan"] = []
            memory["last"] = (7, 0)
            memory["previous_frame"] = obs[:63].copy()
            return "ACTION7", None, None

        choices = []
        for action in model.actions(root):
            if action in banned:
                continue
            result = model.transition(root, action)
            if result is None:
                continue
            nxt, flag = result
            rank = (
                100 if flag == 2 else
                10 + abs(nxt[0] // 16 - py) if nxt[0] // 16 != py else
                2 if action[0] in (3, 4) else 0
            )
            rank -= 10 * repeats.get(action, 0)
            choices.append((rank, action))
        if not choices:
            return reset_or_wait()
        choices.sort(reverse=True)
        plan = [choices[0][1]]

    action = tuple(plan[0])
    result = model.transition(root, action)
    if result is None:
        return reset_or_wait()

    nxt, flag = result
    changes = dict(nxt[2])
    history.append({
        "position": position,
        "gravity": gravity,
        "action": action,
        "before": [(p, world.get(p, UNKNOWN)) for p in changes]
    })
    if len(history) > 160:
        del history[0]

    counts = attempts.setdefault(signature, {})
    counts[action] = counts.get(action, 0) + 1

    world.update(changes)
    memory["expected"] = {
        "position": nxt[0], "gravity": nxt[1], "flag": flag
    }
    memory["plan"] = list(plan[1:]) if flag != 1 else []
    memory["last"] = action
    memory["previous_frame"] = obs[:63].copy()

    kind, target = action
    if kind == 6:
        ty, tx = divmod(target, 16)
        row = top + 6 * (ty - py) + 3
        return (
            "ACTION6",
            int(max(0, min(63, row))),
            int(max(0, min(63, tx * 6 + 3)))
        )
    return "ACTION%d" % kind, None, None

import numpy as np
import heapq
import time
from collections import deque


_DIRS = {
    "ACTION1": (0, -1),
    "ACTION2": (0, 1),
    "ACTION3": (-1, 0),
    "ACTION4": (1, 0),
}
_BASE_DIR = {0: (1, 0), 90: (0, 1),
             180: (-1, 0), 270: (0, -1)}
_PERIMETER = tuple(
    (r, c) for r in range(6) for c in range(6)
    if r in (0, 5) or c in (0, 5)
)
_NOTCHES = {
    0: ((2, 5), (3, 5)),
    90: ((5, 2), (5, 3)),
    180: ((2, 0), (3, 0)),
    270: ((0, 2), (0, 3)),
}
_BAND_PAIRS = {(1, 2), (2, 1), (2, 3), (3, 2)}


def _handles(a):
    result = []
    for y in range(59):
        for x in range(59):
            border = int(a[y, x])
            if border in (0, 1, 2, 3, 4):
                continue
            if (a[y, x + 5] != border or
                    a[y + 5, x] != border or
                    a[y + 5, x + 5] != border):
                continue

            core = int(a[y + 2, x + 2])
            if not (
                a[y + 2, x + 3] == core and
                a[y + 3, x + 2] == core and
                a[y + 3, x + 3] == core
            ):
                continue

            ring = int(a[y + 1, x + 1])
            if not (
                np.all(a[y + 1, x + 1:x + 5] == ring) and
                np.all(a[y + 4, x + 1:x + 5] == ring) and
                np.all(a[y + 2:y + 4, x + 1] == ring) and
                np.all(a[y + 2:y + 4, x + 4] == ring)
            ):
                continue

            normal = core == border and ring in (0, 4)
            locked = border == 5 and ring == 5 and core != 5
            if not (normal or locked):
                continue

            for rotation, notch in _NOTCHES.items():
                pair = tuple(int(a[y + r, x + c]) for r, c in notch)
                if pair not in _BAND_PAIRS:
                    continue
                if all(
                    (r, c) in notch or a[y + r, x + c] == border
                    for r, c in _PERIMETER
                ):
                    result.append((x, y, core, rotation, ring == 0, locked))
                    break
    result.sort(key=lambda h: (h[2], h[3], h[0], h[1]))
    return result


def _boxes(a):
    result = []
    excluded = {0, 1, 2, 3, 4, 5}
    for y in range(61):
        for x in range(61):
            color = int(a[y, x])
            if color in excluded:
                continue
            if (a[y, x + 3] != color or
                    a[y + 3, x] != color or
                    a[y + 3, x + 3] != color):
                continue
            if not (
                np.all(a[y, x:x + 4] == color) and
                np.all(a[y + 3, x:x + 4] == color) and
                np.all(a[y:y + 4, x] == color) and
                np.all(a[y:y + 4, x + 3] == color)
            ):
                continue
            middle = a[y + 1:y + 3, x + 1:x + 3]
            if np.all((middle == color) | (middle == 0)):
                result.append((x - 1, y - 1, color))
    return tuple(sorted(set(result)))


def _bounds(a):
    # Projection estimates remain reliable even when an arm cuts the
    # background into several disconnected components.
    mask = a[:53] == 4
    ys = np.flatnonzero(mask.sum(axis=1) >= 12)
    if len(ys):
        top, bottom = int(ys[0]), int(ys[-1]) + 1
        threshold = max(6, (bottom - top) // 8)
        xs = np.flatnonzero(mask[top:bottom].sum(axis=0) >= threshold)
        if len(xs):
            return int(xs[0]), top, int(xs[-1]) + 1, bottom

    # Conservative fallback for unusually small fields.
    seen = np.zeros(mask.shape, dtype=bool)
    best = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        q = [(int(x), int(y))]
        component = []
        while q:
            xx, yy = q.pop()
            component.append((xx, yy))
            for nx, ny in ((xx - 1, yy), (xx + 1, yy),
                           (xx, yy - 1), (xx, yy + 1)):
                if (0 <= nx < 64 and 0 <= ny < 53 and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    q.append((nx, ny))
        if len(component) > len(best):
            best = component
    if not best:
        return 0, 0, 64, 53
    return (
        min(p[0] for p in best), min(p[1] for p in best),
        max(p[0] for p in best) + 1, max(p[1] for p in best) + 1
    )


def _segment_present(a, x, y, horizontal):
    if not (0 <= x <= 58 and 0 <= y <= 58):
        return False
    if horizontal:
        left = (int(a[y + 2, x]), int(a[y + 3, x]))
        right = (int(a[y + 2, x + 5]), int(a[y + 3, x + 5]))
        return left in ((3, 2), (2, 1)) or right in ((2, 3), (1, 2))
    top = (int(a[y, x + 2]), int(a[y, x + 3]))
    bottom = (int(a[y + 5, x + 2]), int(a[y + 5, x + 3]))
    return top in ((2, 3), (1, 2)) or bottom in ((3, 2), (2, 1))


def _arm_length(a, handle):
    x, y, _, rotation, _, _ = handle
    dx, dy = _BASE_DIR[rotation]
    n = 1
    while n < 16 and _segment_present(
        a, x + 6 * n * dx, y + 6 * n * dy, dx != 0
    ):
        n += 1
    return n


class _Model:
    def __init__(self, a, handles, boxes):
        self.bounds = _bounds(a)
        self.handles = [h for h in handles if h[1] < 53]
        samples = [h for h in handles if h[1] >= 53]
        sample_boxes = {(x, y): c for x, y, c in boxes if y >= 53}

        self.rotations = [h[3] for h in self.handles]
        self.directions = [_BASE_DIR[r] for r in self.rotations]
        self.keys = [(h[2], h[3]) for h in self.handles]
        self.targets = {}
        self.clicks = {}
        self.movable = []

        for i, h in enumerate(self.handles):
            partner = next((s for s in samples if s[2] == h[2]), None)
            if partner is None:
                continue
            sx, sy = partner[:2]
            dx, dy = _BASE_DIR[partner[3]]
            sequence = []
            for k in range(_arm_length(a, partner)):
                color = sample_boxes.get((sx + 6 * k * dx, sy + 6 * k * dy))
                if color is not None:
                    sequence.append(color)
            self.targets[i] = tuple(sequence)
            self.clicks[i] = ("ACTION6", sy + 1, sx + 1)
            self.movable.append(i)

        l, t, r, b = self.bounds

        # Black cells in the field are fixed obstacles. Exclude handle
        # pixels, including the gray outlines of stationary stems.
        black = (a == 5).copy()
        black[:t] = False
        black[b:] = False
        black[:, :l] = False
        black[:, r:] = False
        for x, y, *_ in self.handles:
            black[max(0, y):min(64, y + 6),
                  max(0, x):min(64, x + 6)] = False

        self.obstacles = set()
        visited = set()
        for yy, xx in zip(*np.nonzero(black)):
            p = (int(xx), int(yy))
            if p in visited:
                continue
            visited.add(p)
            q = [p]
            component = []
            while q:
                x, y = q.pop()
                component.append((x, y))
                for nx, ny in ((x - 1, y), (x + 1, y),
                               (x, y - 1), (x, y + 1)):
                    np_ = (nx, ny)
                    if (l <= nx < r and t <= ny < b and
                            black[ny, nx] and np_ not in visited):
                        visited.add(np_)
                        q.append(np_)
            if len(component) >= 9:
                x0 = min(p[0] for p in component)
                x1 = max(p[0] for p in component)
                y0 = min(p[1] for p in component)
                y1 = max(p[1] for p in component)
                self.obstacles.update(
                    (x, y) for y in range(y0, y1 + 1)
                    for x in range(x0, x1 + 1)
                )

        self.rails = {}
        self.configurations = {}

        for i in self.movable:
            h = self.handles[i]
            x, y = h[:2]
            dx, dy = self.directions[i]
            horizontal = dx != 0
            axis = x if horizontal else y
            perp = y if horizontal else x
            low, high = (t + 2, b - 2) if horizontal else (l + 2, r - 2)
            rail = set()

            for p in range(max(0, low), min(64, high)):
                xx, yy = (axis + 2, p) if horizontal else (p, axis + 2)
                if 0 <= xx < 64 and 0 <= yy < 64 and a[yy, xx] in (2, 3):
                    rail.add(p)

            if rail:
                # The movable handle hides the rail underneath it.
                for p in range(max(0, low), min(64, high)):
                    xx, yy = (axis + 2, p) if horizontal else (p, axis + 2)
                    if any(hh[0] <= xx < hh[0] + 6 and
                           hh[1] <= yy < hh[1] + 6
                           for hh in self.handles):
                        rail.add(p)
            self.rails[i] = rail

            reachable = {perp}
            q = deque([perp])
            while q:
                p = q.popleft()
                for sign in (-1, 1):
                    np_ = p + sign * 6
                    if (p + 2 + sign * 3 in rail and
                            -6 <= np_ <= 64 and np_ not in reachable):
                        reachable.add(np_)
                        q.append(np_)

            configs = []
            for p in sorted(reachable):
                bx, by = (axis, p) if horizontal else (p, axis)
                slots = []
                maximum_length = 1
                entered = False
                for k in range(16):
                    xx, yy = bx + 6 * k * dx, by + 6 * k * dy
                    if self.blocked(xx, yy):
                        if entered:
                            break
                        # Normally only segment zero is outside the field.
                        if k > 1:
                            break
                        continue
                    entered = True
                    slots.append((k, xx, yy))
                    maximum_length = k + 1
                if len(slots) >= len(self.targets[i]):
                    configs.append((p, tuple(slots), maximum_length))
            self.configurations[i] = configs

        self.hcache = {}

    def blocked(self, x, y):
        l, t, r, b = self.bounds
        return (x < l or y < t or x + 6 > r or y + 6 > b or
                (x, y) in self.obstacles)

    def read_state(self, a, handles, boxes, previous_active=None):
        candidates = {}
        for h in handles:
            if h[1] < 53:
                candidates.setdefault((h[2], h[3]), []).append(h)
        arms = []
        selected = None
        for i, key in enumerate(self.keys):
            choices = candidates.get(key)
            if not choices:
                return None
            h = choices.pop(0)
            arms.append((h[0], h[1], _arm_length(a, h)))
            if h[4] and i in self.movable:
                selected = i
        if selected is None:
            selected = previous_active
        if selected not in self.movable:
            selected = self.movable[0] if self.movable else 0
        return (
            tuple(arms),
            tuple(sorted((x, y, c) for x, y, c in boxes if y < 53)),
            selected
        )

    def context(self, state):
        arms, boxes, _ = state
        horizontal, vertical = {}, {}
        for i, (x, y, n) in enumerate(arms):
            dx, dy = self.directions[i]
            table = horizontal if dx else vertical
            for k in range(n):
                p = (x + 6 * k * dx, y + 6 * k * dy)
                table.setdefault(p, []).append((i, k))
        box_map = {(x, y): j for j, (x, y, _) in enumerate(boxes)}
        return horizontal, vertical, box_map

    def move(self, state, context, arm_index, action):
        arms, boxes, _ = state
        horizontal, vertical, box_map = context
        x, y, n = arms[arm_index]
        base_dx, base_dy = self.directions[arm_index]
        mx, my = _DIRS[action]
        dx, dy = 6 * mx, 6 * my
        forward = (mx, my) == (base_dx, base_dy)
        backward = (mx, my) == (-base_dx, -base_dy)

        if forward:
            tip_x = x + (n - 1) * 6 * base_dx
            tip_y = y + (n - 1) * 6 * base_dy
            if self.blocked(tip_x + dx, tip_y + dy):
                return None
            indices = range(n)
            new_arm = (x, y, n + 1)
        elif backward:
            if n <= 1:
                return None
            indices = range(1, n)
            new_arm = (x, y, n - 1)
        else:
            perp = y if base_dx else x
            sign = my if base_dx else mx
            if perp + 2 + 3 * sign not in self.rails.get(arm_index, ()):
                return None
            indices = range(n)
            new_arm = (x + dx, y + dy, n)

        moved_boxes = set()
        perpendicular_segments = vertical if mx else horizontal

        def push_box(j, caller=None):
            if j in moved_boxes:
                return True
            bx, by, _ = boxes[j]
            nx, ny = bx + dx, by + dy
            if self.blocked(nx, ny):
                return False
            for p in ((bx, by), (nx, ny)):
                for segment in perpendicular_segments.get(p, ()):
                    if segment != caller:
                        return False
            other = box_map.get((nx, ny))
            if other is not None and not push_box(other):
                return False
            moved_boxes.add(j)
            return True

        for k in indices:
            sx, sy = x + 6 * k * base_dx, y + 6 * k * base_dy
            destination_blocked = self.blocked(sx + dx, sy + dy)
            exempt = backward or self.blocked(sx, sy)
            if destination_blocked and not exempt:
                if forward:
                    continue
                return None
            caller = (arm_index, k)
            for p in ((sx, sy), (sx + dx, sy + dy)):
                j = box_map.get(p)
                if j is not None and not push_box(j, caller):
                    if not (forward or backward):
                        return None

        new_boxes = tuple(sorted(
            (bx + dx, by + dy, color) if j in moved_boxes
            else (bx, by, color)
            for j, (bx, by, color) in enumerate(boxes)
        ))
        new_arms = list(arms)
        new_arms[arm_index] = new_arm
        result = (tuple(new_arms), new_boxes, arm_index)
        return result if result[:2] != state[:2] else None

    def goal(self, state):
        arms, boxes, _ = state
        lookup = {(x, y): c for x, y, c in boxes}
        for i, target in self.targets.items():
            x, y, n = arms[i]
            dx, dy = self.directions[i]
            sequence = []
            for k in range(n):
                c = lookup.get((x + 6 * k * dx, y + 6 * k * dy))
                if c is not None:
                    sequence.append(c)
            if tuple(sequence[:len(target)]) != target:
                return False
        return bool(self.targets)

    def heuristic(self, state):
        key = state[:2]
        cached = self.hcache.get(key)
        if cached is not None:
            return cached

        arms, boxes, _ = state
        by_color = {}
        occupancy = {}
        for x, y, c in boxes:
            by_color.setdefault(c, []).append((x, y))
            occupancy[(x, y)] = c

        total = 0.0
        infinity = 10000.0
        for i, target in self.targets.items():
            count = len(target)
            if not count:
                continue
            if any(c not in by_color for c in target):
                total += 100.0
                continue

            ax, ay, n = arms[i]
            current_perp = ay if self.directions[i][0] else ax
            best = infinity

            for p, slots, maximum_length in self.configurations[i]:
                row_cost = abs(p - current_perp) / 6.0
                shorten = max(0, n - maximum_length)
                dp = [infinity] * count
                dp[0] = 0.0

                for slot_index, sx, sy in slots:
                    occupied = occupancy.get((sx, sy))
                    skip_cost = 1.5 if occupied is not None else 0.0
                    distances = {}
                    for c in set(target):
                        distances[c] = min(
                            (abs(sx - bx) + abs(sy - by)) / 6.0
                            for bx, by in by_color[c]
                        )

                    new_dp = [v + skip_cost for v in dp]
                    for k in range(count):
                        if dp[k] >= infinity:
                            continue
                        c = target[k]
                        wrong = 0.6 if occupied is not None and occupied != c else 0.0
                        box_cost = dp[k] + distances[c] + wrong
                        if k + 1 == count:
                            grow = max(0, slot_index + 1 - n)
                            arm_cost = row_cost + grow + shorten
                            estimate = max(arm_cost, 0.65 * box_cost) + 0.35 * box_cost
                            if estimate < best:
                                best = estimate
                        elif box_cost < new_dp[k + 1]:
                            new_dp[k + 1] = box_cost
                    dp = new_dp
                if best == 0:
                    break
            total += best if best < infinity else 50.0

        if len(self.hcache) > 120000:
            self.hcache.clear()
        self.hcache[key] = total
        return total


def _reconstruct(nodes, index, model):
    chain = []
    while nodes[index][3] is not None:
        state, g, h, parent, operator = nodes[index]
        chain.append((nodes[parent][0], state, operator))
        index = parent
    chain.reverse()

    plan = []
    for before, after, (arm_index, action) in chain:
        if before[2] != arm_index:
            selected = (before[0], before[1], arm_index)
            plan.append((model.clicks[arm_index], selected))
        plan.append(((action, None, None), after))
    return plan


def _search(model, start, available, memory, deadline):
    directions = [a for a in _DIRS if a in available]
    can_click = "ACTION6" in available
    seen_episode = memory.setdefault("seen", {})
    round_number = memory.get("search_round", 0)
    memory["search_round"] = round_number + 1
    weight = (2.8, 1.8, 4.5)[round_number % 3]

    h0 = model.heuristic(start)
    # Nodes: state, path cost, heuristic, parent index, macro operator.
    nodes = [(start, 0, h0, None, None)]
    heap = [(weight * h0, h0, 0, 0)]
    best_cost = {start: 0}
    best_index = None
    best_fallback_score = (float("inf"), float("inf"))
    pops = 0

    while heap:
        if pops % 24 == 0 and time.monotonic() >= deadline:
            break
        _, _, _, index = heapq.heappop(heap)
        state, g, h, _, _ = nodes[index]
        if best_cost.get(state) != g:
            continue
        pops += 1
        if pops > 90000:
            break

        context = model.context(state)
        for arm_index in model.movable:
            switched = arm_index != state[2]
            if switched and not can_click:
                continue
            step_cost = 2 if switched else 1
            for action in directions:
                successor = model.move(state, context, arm_index, action)
                if successor is None:
                    continue
                ng = g + step_cost
                if ng >= best_cost.get(successor, 10**9):
                    continue

                if model.goal(successor):
                    nodes.append((successor, ng, 0.0, index, (arm_index, action)))
                    return _reconstruct(nodes, len(nodes) - 1, model), True

                nh = model.heuristic(successor)
                best_cost[successor] = ng
                node_index = len(nodes)
                nodes.append((successor, ng, nh, index, (arm_index, action)))

                repetitions = seen_episode.get(successor[:2], 0)
                penalty = min(12.0, repetitions * 2.5)
                priority = ng + weight * nh + penalty
                heapq.heappush(heap, (priority, nh, ng, node_index))

                fallback_score = (nh + penalty, ng)
                if fallback_score < best_fallback_score:
                    best_fallback_score = fallback_score
                    best_index = node_index

    if best_index is not None:
        return _reconstruct(nodes, best_index, model), False
    return [], False


def _fallback(available, memory):
    options = [a for a in _DIRS if a in available]
    if options:
        k = memory.get("fallback_counter", 0)
        memory["fallback_counter"] = k + 1
        return options[k % len(options)], None, None
    if "ACTION7" in available:
        return "ACTION7", None, None
    if "RESET" in available:
        return "RESET", None, None
    name = next(iter(available), "ACTION1")
    if name == "ACTION6":
        return name, 0, 0
    return name, None, None


def act(obs, available, memory):
    started = time.monotonic()
    deadline = started + 8.7
    a = np.asarray(obs, dtype=np.int8)
    available = set(available)

    handles = _handles(a)
    boxes = _boxes(a)
    if not handles:
        return _fallback(available, memory)

    model = memory.get("model")
    previous = memory.get("expected")
    previous_active = previous[2] if previous is not None else None

    if model is not None:
        state = model.read_state(a, handles, boxes, previous_active)
    else:
        state = None

    if state is None:
        model = _Model(a, handles, boxes)
        memory["model"] = model
        memory["plan"] = []
        memory["seen"] = {}
        memory["search_round"] = 0
        state = model.read_state(a, handles, boxes)
        if state is None or not model.movable:
            return _fallback(available, memory)

    seen = memory.setdefault("seen", {})
    geometry = state[:2]
    seen[geometry] = seen.get(geometry, 0) + 1

    plan = memory.get("plan", [])
    if previous == state and plan:
        action, after = plan.pop(0)
        if action[0] in available:
            memory["expected"] = after
            memory["plan"] = plan
            return action

    memory["plan"] = []
    plan, solved = _search(model, state, available, memory, deadline)

    if plan:
        action, after = plan.pop(0)
        memory["expected"] = after
        memory["plan"] = plan
        memory["planned_solution"] = solved
        return action

    # If search has no changing move, undo is preferable to blindly
    # spending the movement budget.
    memory["expected"] = None
    if "ACTION7" in available:
        return "ACTION7", None, None
    return _fallback(available, memory)

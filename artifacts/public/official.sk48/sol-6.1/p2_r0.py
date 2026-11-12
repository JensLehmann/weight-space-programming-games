import math
import time
import heapq
import numpy as np


# Directions: up, down, left, right.
_ACTIONS = (
    ("ACTION1", -16, 0, -1),
    ("ACTION2", 16, 0, 1),
    ("ACTION3", -1, -1, 0),
    ("ACTION4", 1, 1, 0),
)
_STEPS = (1, 16, -1, -16)
_DXY = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _rail_present(a, x, y, vertical, hidden=None):
    if not (0 <= x <= 58 and 0 <= y <= 58):
        return False
    if vertical:
        pts = ((x + 2, y), (x + 3, y),
               (x + 2, y + 5), (x + 3, y + 5))
        normal = (2, 3, 3, 2)
        active = (1, 2, 2, 1)
    else:
        pts = ((x, y + 2), (x, y + 3),
               (x + 5, y + 2), (x + 5, y + 3))
        normal = (3, 2, 2, 3)
        active = (2, 1, 1, 2)
    ok1 = ok2 = True
    for k, (px, py) in enumerate(pts):
        if hidden is not None and hidden[py, px]:
            continue
        v = int(a[py, px])
        ok1 = ok1 and v == normal[k]
        ok2 = ok2 and v == active[k]
    return ok1 or ok2


def _find_heads(a):
    roles = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 2],
        [0, 1, 0, 0, 1, 2],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.int8)
    black_roles = np.zeros((6, 6), dtype=np.int8)
    black_roles[2:4, 2:4] = 1
    black_roles[2:4, 5] = 2

    corner = a[:-5, :-5]
    center = a[2:-3, 2:-3]
    inner = a[1:-4, 1:-4]
    corners_equal = (
        (corner == a[:-5, 5:]) &
        (corner == a[5:, :-5]) &
        (corner == a[5:, 5:])
    )
    center_equal = (
        (center == a[2:-3, 3:-2]) &
        (center == a[3:-2, 2:-3]) &
        (center == a[3:-2, 3:-2])
    )

    normal_candidates = (
        corners_equal & center_equal & (center == corner) &
        ((inner == 0) | (inner == 4)) &
        (corner != 0) & (corner != 4)
    )
    black_candidates = (
        corners_equal & center_equal & (corner == 5) &
        (inner == 5) & (center != 5) &
        (center != 0) & (center != 4)
    )

    heads = []
    used = set()
    for kind, candidates, template in (
        (0, normal_candidates, roles),
        (1, black_candidates, black_roles),
    ):
        ys, xs = np.nonzero(candidates)
        for y, x in zip(ys.tolist(), xs.tolist()):
            if (x, y) in used:
                continue
            block = a[y:y + 6, x:x + 6]
            ring = int(block[0, 0])
            color = int(block[2, 2])
            inside = int(block[1, 1])
            choices = []
            for ori in range(4):
                r = np.rot90(template, -ori)
                expected = np.where(r == 1, color if kind else inside,
                                    5 if kind else ring)
                visible = r != 2
                if not np.array_equal(block[visible], expected[visible]):
                    continue
                hole_score = int(np.count_nonzero(
                    block[r == 2] != (5 if kind else ring)))
                dx, dy = _DXY[ori]
                nx, ny = x + 6 * dx, y + 6 * dy
                arm_score = int(_rail_present(a, nx, ny, ori % 2 == 1))
                choices.append((hole_score * 3 + arm_score, ori))
            if choices:
                ori = max(choices)[1]
                heads.append({
                    "x": x, "y": y, "ori": ori, "ring": ring,
                    "color": color, "active": inside == 0,
                    "kind": kind,
                })
                used.add((x, y))
    heads.sort(key=lambda h: (h["y"], h["x"]))
    return heads


def _find_boxes(a):
    c = a[:-3, :-3]
    good = c >= 6
    for dy, dx in (
        (0, 1), (0, 2), (0, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
        (1, 0), (2, 0), (1, 3), (2, 3),
    ):
        good = good & (a[dy:dy + 61, dx:dx + 61] == c)
    middle_same = np.ones_like(good)
    middle_zero = np.ones_like(good)
    for dy, dx in ((1, 1), (1, 2), (2, 1), (2, 2)):
        z = a[dy:dy + 61, dx:dx + 61]
        middle_same &= z == c
        middle_zero &= z == 0
    good &= middle_same | middle_zero

    boxes = []
    ys, xs = np.nonzero(good)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if x >= 1 and y >= 1:
            boxes.append((x - 1, y - 1, int(a[y, x])))
    return boxes


def _find_guides(a, hidden, bx, by, width, height):
    guide = np.zeros((64, 64), dtype=bool)
    pattern = np.array([2, 2, 3, 3, 3, 3, 2, 2], dtype=np.int8)

    # A guide has width two. Heads can obscure part of its eight-pixel tiles.
    for x in range(63):
        if bx <= x < bx + width:
            continue
        for y in range(46):
            b = a[y:y + 8, x:x + 2]
            mask = ~hidden[y:y + 8, x:x + 2]
            if np.count_nonzero(mask) < 4:
                continue
            expected = np.broadcast_to(pattern[:, None], (8, 2))
            if np.all(b[mask] == expected[mask]):
                guide[y:y + 8, x:x + 2] = True

    for y in range(53):
        if by <= y < by + height or y > 51:
            continue
        for x in range(57):
            b = a[y:y + 2, x:x + 8]
            mask = ~hidden[y:y + 2, x:x + 8]
            if np.count_nonzero(mask) < 4:
                continue
            expected = np.broadcast_to(pattern[None, :], (2, 8))
            if np.all(b[mask] == expected[mask]):
                guide[y:y + 2, x:x + 8] = True
    return guide


class _Model:
    def enc(self, x, y):
        u = int(round((x - self.bx) / 6.0)) + 2
        v = int(round((y - self.by) / 6.0)) + 2
        return u + 16 * v

    def xy(self, p):
        return self.bx + 6 * (p % 16 - 2), self.by + 6 * (p // 16 - 2)

    def blocked(self, p):
        u, v = p % 16, p // 16
        return not (2 <= u < 2 + self.w and
                    2 <= v < 2 + self.h) or p in self.walls

    def guide_move(self, p, delta):
        x, y = self.xy(p)
        if delta == 1:
            dx, dy = 1, 0
        elif delta == -1:
            dx, dy = -1, 0
        elif delta == 16:
            dx, dy = 0, 1
        else:
            dx, dy = 0, -1
        sx, sy = x + 2 + 3 * dx, y + 2 + 3 * dy
        return (0 <= sx < 64 and 0 <= sy < 53 and
                bool(self.guide[sy, sx]))

    def rail_maps(self, heads, removing=None):
        horizontal = dict(self.fixed_h)
        vertical = dict(self.fixed_v)
        for i, meta in enumerate(self.controls):
            p, n = heads[2 * i:2 * i + 2]
            table = vertical if meta["ori"] % 2 else horizontal
            step = meta["step"]
            first = 1 if removing == i else 0
            for k in range(first, n):
                q = p + k * step
                table[q] = table.get(q, 0) + 1
        return horizontal, vertical

    def transition(self, state, controller, delta):
        heads, boxes, selected = state
        p, n = heads[2 * controller:2 * controller + 2]
        meta = self.controls[controller]
        base = meta["step"]

        extending = delta == base
        retracting = delta == -base
        lateral = not extending and not retracting

        if extending:
            if self.blocked(p + n * base):
                return None
        elif retracting:
            if n <= 1:
                return None
        elif not self.guide_move(p, delta):
            return None

        horizontal, vertical = self.rail_maps(
            heads, controller if retracting else None)
        lookup = {q: j for j, q in enumerate(boxes)}
        movers = set()
        moving_vertical = abs(delta) == 16
        segment_vertical = meta["ori"] % 2 == 1
        perpendicular = horizontal if moving_vertical else vertical

        def move_box(j, caller_position, caller_vertical):
            if j in movers:
                return True
            q = boxes[j]
            target = q + delta
            if self.blocked(target):
                return False

            # A block cannot move across a perpendicular rail, except for
            # the segment currently carrying that block.
            for r in (q, target):
                count = perpendicular.get(r, 0)
                if (caller_vertical != moving_vertical and
                        r == caller_position):
                    count -= 1
                if count > 0:
                    return False

            other = lookup.get(target)
            if other is not None:
                if not move_box(other, caller_position, caller_vertical):
                    return False
            movers.add(j)
            return True

        first = 1 if retracting else 0
        for k in range(first, n):
            q = p + k * base
            target = q + delta
            if self.blocked(target):
                if not (retracting or self.blocked(q)):
                    if lateral:
                        return None
                    continue

            for r in (q, target):
                j = lookup.get(r)
                if j is not None:
                    if not move_box(j, q, segment_vertical) and lateral:
                        return None

        new_boxes = list(boxes)
        for j in movers:
            new_boxes[j] += delta
        for start, end in self.groups:
            if end - start > 1:
                new_boxes[start:end] = sorted(new_boxes[start:end])

        new_heads = list(heads)
        if extending:
            new_heads[2 * controller + 1] = n + 1
        elif retracting:
            new_heads[2 * controller + 1] = n - 1
        else:
            new_heads[2 * controller] = p + delta

        return tuple(new_heads), tuple(new_boxes), controller

    def goal(self, state):
        heads, boxes, selected = state
        lookup = {p: self.colors[j] for j, p in enumerate(boxes)}
        for i, meta in enumerate(self.controls):
            target = meta["target"]
            if not target:
                continue
            p, n = heads[2 * i:2 * i + 2]
            found = 0
            for k in range(n):
                c = lookup.get(p + k * meta["step"])
                if c is not None:
                    if c != target[found]:
                        return False
                    found += 1
                    if found == len(target):
                        break
            if found < len(target):
                return False
        return True

    def make_options(self, initial_heads):
        self.options = []
        for i, meta in enumerate(self.controls):
            initial = initial_heads[2 * i]
            lateral_steps = (-16, 16) if meta["ori"] % 2 == 0 else (-1, 1)
            reached = {initial}
            todo = [initial]
            while todo and len(reached) < 64:
                p = todo.pop()
                for delta in lateral_steps:
                    q = p + delta
                    x, y = self.xy(q)
                    if not (0 <= x <= 58 and 0 <= y <= 58):
                        continue
                    if q not in reached and self.guide_move(p, delta):
                        reached.add(q)
                        todo.append(q)

            options = []
            for p in sorted(reached):
                cells = []
                for k in range(13):
                    q = p + k * meta["step"]
                    if self.blocked(q):
                        if k == 0:
                            continue
                        break
                    cells.append((self.cell_index[q], k + 1))
                if len(cells) >= len(meta["target"]):
                    options.append((p, tuple(cells)))
            self.options.append(options)

    def profiles(self, boxes):
        cached = self.profile_cache.get(boxes)
        if cached is not None:
            return cached

        occupants = {p: self.colors[j] for j, p in enumerate(boxes)}
        near = {}
        required = set()
        for meta in self.controls:
            required.update(meta["target"])

        for color in required:
            indices = self.color_indices.get(color, ())
            if not indices:
                near[color] = (100.0,) * len(self.cells)
            elif len(indices) == 1:
                near[color] = self.distances[boxes[indices[0]]]
            else:
                arrays = [self.distances[boxes[j]] for j in indices]
                near[color] = tuple(min(values) for values in zip(*arrays))

        result = []
        for i, meta in enumerate(self.controls):
            target = meta["target"]
            count = len(target)
            rail_profiles = []
            if count == 0:
                result.append(rail_profiles)
                continue

            for head_position, cells in self.options[i]:
                dp = [0.0] + [1e6] * (count - 1)
                completions = []
                for number, (cell_index, required_length) in enumerate(cells):
                    q = self.cells[cell_index]
                    occupant = occupants.get(q)
                    skip = 2.0 if occupant is not None else 0.0

                    if number >= count - 1:
                        color = target[-1]
                        wrong = 1.0 if (occupant is not None and
                                        occupant != color) else 0.0
                        cost = dp[count - 1] + near[color][cell_index] + wrong
                        completions.append((required_length, cost))

                    for j in range(min(count - 1, number + 1), 0, -1):
                        color = target[j - 1]
                        wrong = 1.0 if (occupant is not None and
                                        occupant != color) else 0.0
                        take = dp[j - 1] + near[color][cell_index] + wrong
                        dp[j] = min(dp[j] + skip, take)
                    dp[0] += skip

                rail_profiles.append((head_position, tuple(completions)))
            result.append(rail_profiles)

        if len(self.profile_cache) >= 3500:
            self.profile_cache.clear()
        self.profile_cache[boxes] = result
        return result

    def heuristic(self, state):
        heads, boxes, selected = state
        key = (heads, boxes)
        cached = self.h_cache.get(key)
        if cached is not None:
            return cached

        profiles = self.profiles(boxes)
        total = 0.0
        for i, meta in enumerate(self.controls):
            if not meta["target"]:
                continue
            p, n = heads[2 * i:2 * i + 2]
            best = 1e5
            for target_position, completions in profiles[i]:
                displacement = (
                    abs(p % 16 - target_position % 16) +
                    abs(p // 16 - target_position // 16)
                )
                move_cost = 1.15 * displacement
                for required_length, arrangement_cost in completions:
                    cost = (arrangement_cost + move_cost +
                            max(0, required_length - n))
                    if cost < best:
                        best = cost
            total += best

        if len(self.h_cache) >= 100000:
            self.h_cache.clear()
        self.h_cache[key] = total
        return total

    def validate(self, a, state):
        heads, boxes, selected = state
        hidden = np.zeros((64, 64), dtype=bool)
        for h in self.static_head_images:
            x, y = h["x"], h["y"]
            hidden[y:y + 6, x:x + 6] = True

        for i, meta in enumerate(self.controls):
            p, n = heads[2 * i:2 * i + 2]
            x, y = self.xy(p)
            if not (0 <= x <= 58 and 0 <= y <= 58):
                return False
            if int(a[y, x]) != meta["ring"]:
                return False
            if int(a[y + 2, x + 2]) != meta["color"]:
                return False
            if int(a[y + 1, x + 1]) != (0 if selected == i else 4):
                return False
            hidden[y:y + 6, x:x + 6] = True

        for j, p in enumerate(boxes):
            x, y = self.xy(p)
            if not (0 <= x <= 58 and 0 <= y <= 58):
                return False
            c = self.colors[j]
            if not (
                int(a[y + 1, x + 1]) == c and
                int(a[y + 1, x + 4]) == c and
                int(a[y + 4, x + 1]) == c and
                int(a[y + 4, x + 4]) == c
            ):
                return False

        for i, meta in enumerate(self.controls):
            p, n = heads[2 * i:2 * i + 2]
            for k in range(1, n):
                x, y = self.xy(p + k * meta["step"])
                if not _rail_present(a, x, y, meta["ori"] % 2 == 1, hidden):
                    return False
            x, y = self.xy(p + n * meta["step"])
            if _rail_present(a, x, y, meta["ori"] % 2 == 1, hidden):
                return False
        return True


def _parse(a):
    heads = _find_heads(a)
    boxes = _find_boxes(a)
    if not heads:
        return None, None

    hidden = np.zeros((64, 64), dtype=bool)
    for h in heads:
        hidden[h["y"]:h["y"] + 6, h["x"]:h["x"] + 6] = True

    board = (a == 4).copy()
    board[53:] = False
    board[hidden] = False
    ys, xs = np.nonzero(board)
    if len(xs) == 0:
        return None, None
    bx, by = int(xs.min()), int(ys.min())
    width = int(xs.max()) - bx + 1
    height = int(ys.max()) - by + 1
    if width < 6 or height < 6:
        return None, None

    m = _Model()
    m.bx, m.by = bx, by
    m.w = max(1, int(round(width / 6.0)))
    m.h = max(1, int(round(height / 6.0)))
    if m.w > 10 or m.h > 10:
        return None, None
    m.guide = _find_guides(a, hidden, bx, by, width, height)

    lengths = {}
    for index, h in enumerate(heads):
        dx, dy = _DXY[h["ori"]]
        n = 1
        while n < 13:
            x, y = h["x"] + 6 * dx * n, h["y"] + 6 * dy * n
            if not _rail_present(a, x, y, h["ori"] % 2 == 1, hidden):
                break
            n += 1
        lengths[index] = n

    box_at = {(x, y): c for x, y, c in boxes}
    bottom = [i for i, h in enumerate(heads) if h["y"] >= 53]
    controls = []
    controlled_indices = set()
    initial_heads = []
    selected = 0

    for index, h in enumerate(heads):
        if h["y"] >= 53 or h["kind"] != 0:
            continue
        partner = next(
            (j for j in bottom if heads[j]["color"] == h["color"]), None)
        if partner is None:
            continue

        b = heads[partner]
        dx, dy = _DXY[b["ori"]]
        target = []
        for k in range(lengths[partner]):
            c = box_at.get((b["x"] + 6 * dx * k, b["y"] + 6 * dy * k))
            if c is not None:
                target.append(c)

        meta = dict(h)
        meta["step"] = _STEPS[h["ori"]]
        meta["target"] = tuple(target)
        if h["active"]:
            selected = len(controls)
        controls.append(meta)
        controlled_indices.add(index)
        initial_heads.extend((m.enc(h["x"], h["y"]), lengths[index]))

    if not controls:
        return None, None
    m.controls = controls
    m.static_head_images = [
        h for i, h in enumerate(heads) if i not in controlled_indices
    ]

    m.fixed_h = {}
    m.fixed_v = {}
    for i, h in enumerate(heads):
        if h["y"] >= 53 or i in controlled_indices:
            continue
        p = m.enc(h["x"], h["y"])
        step = _STEPS[h["ori"]]
        table = m.fixed_v if h["ori"] % 2 else m.fixed_h
        for k in range(lengths[i]):
            q = p + k * step
            table[q] = table.get(q, 0) + 1

    m.walls = set()
    for v in range(m.h):
        for u in range(m.w):
            x, y = bx + 6 * u, by + 6 * v
            tile = a[y:y + 6, x:x + 6]
            if tile.shape == (6, 6) and np.count_nonzero(tile == 5) >= 30:
                m.walls.add((u + 2) + 16 * (v + 2))

    world_boxes = sorted(
        (c, m.enc(x, y)) for x, y, c in boxes if y < 53
    )
    m.colors = tuple(c for c, p in world_boxes)
    initial_boxes = tuple(p for c, p in world_boxes)
    m.groups = []
    m.color_indices = {}
    start = 0
    while start < len(world_boxes):
        color = world_boxes[start][0]
        end = start + 1
        while end < len(world_boxes) and world_boxes[end][0] == color:
            end += 1
        m.groups.append((start, end))
        m.color_indices[color] = tuple(range(start, end))
        start = end

    m.cells = tuple(
        (u + 2) + 16 * (v + 2)
        for v in range(m.h) for u in range(m.w)
    )
    m.cell_index = {p: j for j, p in enumerate(m.cells)}
    m.distances = {}
    for p in m.cells:
        m.distances[p] = tuple(
            abs(p % 16 - q % 16) + abs(p // 16 - q // 16)
            for q in m.cells
        )
    # Usually all blocks are inside the board; retain support for other starts.
    for p in initial_boxes:
        if p not in m.distances:
            m.distances[p] = tuple(
                abs(p % 16 - q % 16) + abs(p // 16 - q // 16)
                for q in m.cells
            )

    m.profile_cache = {}
    m.h_cache = {}
    initial_heads = tuple(initial_heads)
    m.make_options(initial_heads)
    return m, (initial_heads, initial_boxes, selected)


def _build_plan(m, root, end, parents):
    path = []
    state = end
    while state != root:
        previous, controller, action = parents[state]
        path.append((previous, controller, action, state))
        state = previous
    path.reverse()

    plan = []
    for previous, controller, action, following in path:
        if previous[2] != controller:
            p = previous[0][2 * controller]
            x, y = m.xy(p)
            clicked = (previous[0], previous[1], controller)
            plan.append(("ACTION6", y + 1, x + 1, clicked))
        plan.append((action, None, None, following))
    return plan


def _search(m, root, allowed, history, seconds=7.5):
    deadline = time.perf_counter() + seconds
    parents = {}
    costs = {root: 0.0}
    root_h = m.heuristic(root)
    queue = [(2.8 * root_h, root_h, 0, 0.0, root)]
    serial = 1
    expansions = 0
    best_partial = None
    best_partial_score = float("inf")
    solution = None

    directions = [item for item in _ACTIONS if item[0] in allowed]
    can_click = "ACTION6" in allowed

    while queue and len(costs) < 120000:
        if expansions % 64 == 0 and time.perf_counter() >= deadline:
            break
        _, _, _, g, state = heapq.heappop(queue)
        if g != costs.get(state):
            continue
        expansions += 1

        for controller in range(len(m.controls)):
            if controller != state[2] and not can_click:
                continue
            switching = controller != state[2]
            for action, delta, _, _ in directions:
                following = m.transition(state, controller, delta)
                if following is None:
                    continue
                step_cost = 2.0 if switching else 1.0
                repetition = history.get(following, 0)
                ng = g + step_cost + 4.0 * repetition
                if ng >= costs.get(following, float("inf")):
                    continue

                costs[following] = ng
                parents[following] = (state, controller, action)
                if m.goal(following):
                    solution = following
                    break

                nh = m.heuristic(following)
                score = nh + 0.025 * ng + 0.5 * repetition
                if score < best_partial_score:
                    best_partial_score = score
                    best_partial = following
                heapq.heappush(
                    queue, (ng + 2.8 * nh, nh, serial, ng, following))
                serial += 1
            if solution is not None:
                break
        if solution is not None:
            break

    end = solution if solution is not None else best_partial
    if end is None:
        return []
    plan = _build_plan(m, root, end, parents)
    if solution is None:
        # Replan after making a little progress rather than committing to a
        # long, unfinished search branch.
        plan = plan[:6]
    return plan


def act(obs, available, memory):
    a = np.asarray(obs, dtype=np.int8)
    allowed = set(available)

    def fallback():
        if "RESET" in allowed:
            return "RESET", None, None
        if "ACTION7" in allowed:
            return "ACTION7", None, None
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if name in allowed:
                return name, None, None
        if "ACTION6" in allowed:
            return "ACTION6", 0, 0
        return (available[0], None, None) if available else ("RESET", None, None)

    if not np.any(a[53] == 2) and "RESET" in allowed:
        memory.clear()
        return "RESET", None, None

    model = memory.get("model")
    state = memory.get("predicted")
    valid = False
    if model is not None and state is not None:
        try:
            valid = model.validate(a, state)
        except (KeyError, IndexError, ValueError):
            valid = False

    if not valid:
        try:
            model, state = _parse(a)
        except (KeyError, IndexError, ValueError, TypeError):
            model, state = None, None
        memory["plan"] = []
        if model is None:
            memory.clear()
            return fallback()
        memory["model"] = model
        memory["predicted"] = state

    history = memory.setdefault("history", {})
    history[state] = history.get(state, 0) + 1

    plan = memory.get("plan", [])
    if plan and plan[0][0] not in allowed:
        plan = []
    if not plan:
        try:
            plan = _search(model, state, allowed, history)
        except (KeyError, IndexError, ValueError, RecursionError):
            plan = []

    if not plan:
        # Prefer a legal modelled move if search could not supply a plan.
        candidates = []
        for controller in range(len(model.controls)):
            if controller != state[2] and "ACTION6" not in allowed:
                continue
            for action, delta, _, _ in _ACTIONS:
                if action not in allowed:
                    continue
                following = model.transition(state, controller, delta)
                if following is not None:
                    candidates.append((
                        model.heuristic(following) +
                        4 * history.get(following, 0),
                        controller, action, following
                    ))
        if candidates:
            _, controller, action, following = min(
                candidates, key=lambda item: item[0])
            if controller != state[2]:
                x, y = model.xy(state[0][2 * controller])
                clicked = (state[0], state[1], controller)
                plan = [
                    ("ACTION6", y + 1, x + 1, clicked),
                    (action, None, None, following),
                ]
            else:
                plan = [(action, None, None, following)]
        else:
            memory["plan"] = []
            return fallback()

    action, row, col, following = plan[0]
    memory["plan"] = plan[1:]
    memory["predicted"] = following
    return action, row, col

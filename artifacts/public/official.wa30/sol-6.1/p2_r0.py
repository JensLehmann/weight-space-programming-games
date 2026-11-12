import numpy as np
import heapq
import itertools
import time
from collections import deque

INF = 10**6
DX = (0, 0, -1, 1)
DY = (-1, 1, 0, 0)

NEI = []
for p in range(256):
    x, y = p % 16, p // 16
    NEI.append(tuple(
        (y + dy) * 16 + x + dx
        if 0 <= x + dx < 16 and 0 <= y + dy < 16 else -1
        for dx, dy in zip(DX, DY)
    ))

AUTO_ORDER = (2, 3, 0, 1)


def adj(a, b):
    return a >= 0 and b >= 0 and b in NEI[a]


def prefix(a):
    return np.pad(a.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def rectsum(a, x0, y0, x1, y1):
    return int(a[y1, x1] - a[y0, x1] - a[y1, x0] + a[y0, x0])


def ground_rectangles(obs, ignored, goals, forbidden):
    known = ~ignored.copy()
    known[63] = False
    if goals:
        bad9 = prefix(known & (obs != 9))
        bad2 = prefix(known & (obs != 2))
        support9 = prefix(known & (obs == 9))
        excluded = prefix(forbidden)
    else:
        bad2 = prefix((known & (obs != 2)) | forbidden)
        support2 = prefix(known & (obs == 2) & ~forbidden)

    candidates = []
    for y0 in range(0, 64, 4):
        for y1 in range(y0 + 4, 65, 4):
            for x0 in range(0, 64, 4):
                for x1 in range(x0 + 4, 65, 4):
                    if goals:
                        if rectsum(excluded, x0, y0, x1, y1):
                            continue
                        outer = rectsum(bad9, x0, y0, x1, y1)
                        inner = rectsum(bad9, x0 + 1, y0 + 1, x1 - 1, y1 - 1)
                        if outer != inner:
                            continue
                        if rectsum(bad2, x0 + 1, y0 + 1, x1 - 1, y1 - 1):
                            continue
                        support = rectsum(support9, x0, y0, x1, y1)
                        if support < 4:
                            continue
                    else:
                        if rectsum(bad2, x0, y0, x1, y1):
                            continue
                        support = rectsum(support2, x0, y0, x1, y1)
                        if not support:
                            continue
                    area = (x1 - x0) * (y1 - y0)
                    candidates.append((support, area, x0, y0, x1, y1))

    candidates.sort(reverse=True)
    result = np.zeros((64, 64), dtype=bool)
    for _, _, x0, y0, x1, y1 in candidates:
        if not result[y0:y1, x0:x1].any():
            result[y0:y1, x0:x1] = True
    return result


def match_positions(old, new):
    if old is None:
        return tuple(sorted(new))
    result = [-1] * len(old)
    remaining = set(new)
    for i, p in enumerate(old):
        if p >= 0 and p in remaining:
            result[i] = p
            remaining.remove(p)
    for i, p in enumerate(old):
        if result[i] >= 0 or p < 0 or not remaining:
            continue
        q = min(remaining, key=lambda q:
                abs(q % 16 - p % 16) + abs(q // 16 - p // 16))
        result[i] = q
        remaining.remove(q)
    result.extend(sorted(remaining))
    return tuple(result)


def read_board(obs, memory):
    boxes, allies, enemies = [], [], []
    borders = {}
    player, rotation = None, None
    ignored = np.zeros((64, 64), dtype=bool)
    walls, hazards = set(), set()

    for p in range(256):
        y, x = p // 16 * 4, p % 16 * 4
        tile = obs[y:y + 4, x:x + 4]
        c = int(tile[1, 1])
        sprite = False

        if c == 9 and int(tile[2, 2]) == 9:
            boxes.append(p)
            borders[p] = int(tile[0, 0])
            sprite = True
        elif c == 12:
            allies.append(p)
            sprite = True
        elif c == 15:
            enemies.append(p)
            sprite = True
        elif c == 14:
            player = p
            sprite = True
            if np.all(tile[0] == 0):
                rotation = 0
            elif np.all(tile[:3, 3] == 0):
                rotation = 3
            elif np.all(tile[:3, 0] == 0):
                rotation = 2
            else:
                rotation = 1

        if sprite:
            ignored[y:y + 4, x:x + 4] = True
        else:
            visible = tile[:3] if y == 60 else tile
            if np.all(visible == 5):
                walls.add(p)
            if np.any(visible == 2) and np.any(visible == 1):
                hazards.add(p)

    expected = memory.get("expected")
    if player is None:
        player = expected[0] if expected else 0
    if rotation is None:
        rotation = expected[1] if expected else 0

    if "env" not in memory:
        forbidden = np.zeros((64, 64), dtype=bool)
        for p in walls | hazards:
            y, x = p // 16 * 4, p % 16 * 4
            forbidden[y:y + 4, x:x + 4] = True

        gp = ground_rectangles(obs, ignored, True, forbidden)
        rp = ground_rectangles(obs, ignored, False, forbidden | gp)
        goals, red = [], []
        for p in range(256):
            y, x = p // 16 * 4, p % 16 * 4
            if gp[y:y + 4, x:x + 4].any():
                goals.append(p)
            if rp[y:y + 4, x:x + 4].any():
                red.append(p)

        # Fill obscured straight sections of patterned barriers.
        for p in boxes + allies + enemies + [player]:
            n = NEI[p]
            if ((n[0] in hazards and n[1] in hazards)
                    or (n[2] in hazards and n[3] in hazards)):
                if p not in goals and p not in red:
                    hazards.add(p)

        memory["env"] = {
            "walls": tuple(walls),
            "hazards": tuple(hazards),
            "goals": tuple(goals),
            "red": tuple(red),
        }

    bt = match_positions(expected[4] if expected else None, boxes)
    if expected:
        na = memory["na"]
        ap = match_positions(expected[2][:na], allies)
        ep = match_positions(expected[2][na:], enemies)
    else:
        ap, ep = tuple(sorted(allies)), tuple(sorted(enemies))
        memory["na"] = len(ap)

    actors = ap + ep
    positions = (player,) + actors
    holds = [-1] * len(positions)

    for b, q in enumerate(bt):
        if borders.get(q) == 0 and adj(player, q):
            holds[0] = b

    used = {b for b in holds if b >= 0}
    if expected:
        for i, b in enumerate(expected[3]):
            if i == 0 or i >= len(holds) or not 0 <= b < len(bt):
                continue
            if (b not in used and borders.get(bt[b]) in (3, 5)
                    and adj(positions[i], bt[b])):
                holds[i] = b
                used.add(b)

    previous = memory.get("last_state")
    if previous:
        for i, b in enumerate(previous[3]):
            if i == 0 or i >= len(holds) or not 0 <= b < len(bt):
                continue
            if (holds[i] < 0 and b not in used
                    and borders.get(bt[b]) == 3
                    and adj(positions[i], bt[b])):
                holds[i] = b
                used.add(b)

    for b, q in enumerate(bt):
        if b in used or borders.get(q) != 5:
            continue
        candidates = [
            i for i in range(1, len(holds))
            if holds[i] < 0 and adj(positions[i], q)
        ]
        if candidates:
            holds[candidates[0]] = b
            used.add(b)

    if expected is None:
        for b, q in enumerate(bt):
            if b in used or borders.get(q) != 3:
                continue
            candidates = [
                i for i in range(1, len(holds))
                if holds[i] < 0 and adj(positions[i], q)
            ]
            if len(candidates) == 1:
                holds[candidates[0]] = b
                used.add(b)

    return player, rotation, actors, tuple(holds), bt


class Model:
    def __init__(self, env, na, memory):
        self.walls = set(env["walls"])
        self.hazards = set(env["hazards"])
        self.goals = set(env["goals"])
        self.red = set(env["red"])
        self.na = na
        self.order_a = tuple(memory.get("order_a", range(na)))
        self.order_e = tuple(memory.get("order_e", range(memory.get("ne", 0))))
        self.priority = tuple(memory.get("priority", ()))
        self.cache = {}
        self.walk_cache = {}
        self.shape_cache = {}
        self.protected = set()

    @staticmethod
    def shifted(p, offset):
        x, y = p % 16 + offset[0], p // 16 + offset[1]
        return y * 16 + x if 0 <= x < 16 and 0 <= y < 16 else -1

    def occupied(self, p, actors, boxes):
        return self.walls | {p} | {a for a in actors if a >= 0} | set(boxes)

    def fixed_boxes(self, state, except_box=-1):
        return frozenset(
            q for b, q in enumerate(state[4])
            if b != except_box and q in self.goals
        )

    def component(self, start):
        seen = {start}
        queue = deque([start])
        blocked = self.walls | self.hazards
        while queue:
            p = queue.popleft()
            for q in NEI[p]:
                if q >= 0 and q not in blocked and q not in seen:
                    seen.add(q)
                    queue.append(q)
        return seen

    def walk_dist(self, start, fixed=frozenset()):
        key = start, fixed
        if key in self.walk_cache:
            return self.walk_cache[key]
        dist = [INF] * 256
        blocked = self.walls | set(fixed)
        if start >= 0:
            dist[start] = 0
            queue = deque([start])
            while queue:
                p = queue.popleft()
                for q in NEI[p]:
                    if (q >= 0 and q not in blocked
                            and q not in self.hazards and dist[q] == INF):
                        dist[q] = dist[p] + 1
                        queue.append(q)
        self.walk_cache[key] = dist
        return dist

    def shape_dist(self, offset, fixed=frozenset()):
        key = offset, fixed
        if key in self.shape_cache:
            return self.shape_cache[key]
        blocked = self.walls | set(fixed)
        valid = [False] * 256
        dist = [INF] * 256
        queue = deque()
        for p in range(256):
            b = self.shifted(p, offset)
            valid[p] = (
                p not in blocked and p not in self.hazards
                and b >= 0 and b not in blocked
            )
            if valid[p] and b in self.goals:
                dist[p] = 0
                queue.append(p)
        while queue:
            p = queue.popleft()
            for q in NEI[p]:
                if q >= 0 and valid[q] and dist[q] == INF:
                    dist[q] = dist[p] + 1
                    queue.append(q)
        self.shape_cache[key] = dist
        return dist

    def path_step(self, start, occupied, targets=None,
                  box=None, destination=None):
        if start < 0:
            return None
        seen = {start}
        queue = deque([(start, -1)])
        if box is not None:
            offset = box % 16 - start % 16, box // 16 - start // 16
        while queue:
            p, first = queue.popleft()
            reached = (
                p in targets if box is None
                else self.shifted(p, offset) in destination
            )
            if reached:
                return start if first < 0 else first
            for k in AUTO_ORDER:
                q = NEI[p][k]
                if q < 0 or q in seen or q in self.hazards:
                    continue
                if box is None:
                    if q in occupied:
                        continue
                else:
                    bq = self.shifted(q, offset)
                    if q in occupied and q != box:
                        continue
                    if bq < 0 or (bq in occupied and bq != start):
                        continue
                seen.add(q)
                queue.append((q, q if first < 0 else first))
        return None

    def step(self, state, action):
        key = state, action
        if key in self.cache:
            return self.cache[key]

        p, rotation, at, ht, bt = state
        actors, holds, boxes = list(at), list(ht), list(bt)

        def occupancy():
            return self.occupied(p, actors, boxes)

        def move(i, q):
            nonlocal p
            old = p if i == 0 else actors[i - 1]
            occupied = occupancy()
            if q < 0 or q in self.hazards:
                return False
            b = holds[i]
            if b < 0:
                if q in occupied:
                    return False
            else:
                bp = boxes[b]
                offset = bp % 16 - old % 16, bp // 16 - old // 16
                bq = self.shifted(q, offset)
                if q in occupied and q != bp:
                    return False
                if bq < 0 or (bq in occupied and bq != old):
                    return False
                boxes[b] = bq
            if i == 0:
                p = q
            else:
                actors[i - 1] = q
            return True

        def attach(i, b):
            for j in range(len(holds)):
                if holds[j] == b:
                    holds[j] = -1
            holds[i] = b

        if 1 <= action <= 4:
            if holds[0] < 0:
                rotation = action - 1
            move(0, NEI[p][action - 1])
        elif action == 5:
            if holds[0] >= 0:
                holds[0] = -1
            else:
                front = NEI[p][rotation]
                for b in self.priority or range(len(boxes)):
                    if b < len(boxes) and boxes[b] == front:
                        attach(0, b)
                        break
                for j in range(self.na, len(actors)):
                    if front >= 0 and actors[j] == front:
                        holds[j + 1] = -1
                        actors[j] = -1
                        break

        def automate(indices, friendly):
            destination = self.goals if friendly else self.red
            for j in indices:
                if j >= len(actors) or actors[j] < 0:
                    continue
                i, a = j + 1, actors[j]
                b = holds[i]
                if b >= 0:
                    if boxes[b] in destination:
                        holds[i] = -1
                    else:
                        q = self.path_step(
                            a, occupancy(), box=boxes[b],
                            destination=destination
                        )
                        if q is not None and q != a:
                            move(i, q)
                    continue

                eligible = []
                for k in self.priority or range(len(boxes)):
                    if k >= len(boxes) or boxes[k] in destination:
                        continue
                    owner = next((z for z, v in enumerate(holds) if v == k), -1)
                    if (owner < 0 if friendly else owner <= self.na):
                        eligible.append(k)

                for k in eligible:
                    if adj(a, boxes[k]):
                        attach(i, k)
                        return

                targets = set()
                for k in eligible:
                    targets.update(q for q in NEI[boxes[k]] if q >= 0)
                q = self.path_step(a, occupancy(), targets=targets)
                if q is not None and q != a:
                    move(i, q)

        automate(self.order_a, True)
        automate(tuple(self.na + j for j in self.order_e), False)
        result = p, rotation, tuple(actors), tuple(holds), tuple(boxes)
        self.cache[key] = result
        return result

    def delivered(self, state, b):
        return state[4][b] in self.goals and b not in state[3]

    def won(self, state):
        return all(self.delivered(state, b) for b in range(len(state[4])))

    def viable_carrier(self, state, b):
        owner = next((i for i, k in enumerate(state[3]) if k == b), -1)
        if not 0 < owner <= self.na:
            return False
        a = state[2][owner - 1]
        return self.path_step(
            a, self.occupied(state[0], state[2], state[4]),
            box=state[4][b], destination=self.goals
        ) is not None

    def enemy_h(self, state):
        d = self.walk_dist(state[0])
        return min(
            (d[q] + 1 for a in state[2][self.na:] if a >= 0
             for q in NEI[a] if q >= 0),
            default=INF
        )

    @staticmethod
    def reconstruct(parents, state, start):
        actions = []
        while state != start:
            state, action = parents[state]
            actions.append(action)
        return actions[::-1]

    def intercept(self, state, deadline):
        living = sum(a >= 0 for a in state[2][self.na:])
        h0 = self.enemy_h(state)
        if not living or h0 >= INF:
            return None
        heap = [(2.2 * h0, 0, 0, state)]
        costs, parents = {state: 0}, {}
        serial, expanded = 0, 0
        best, best_h = state, h0

        while heap and expanded < 2600 and time.monotonic() < deadline:
            _, g, _, s = heapq.heappop(heap)
            if costs.get(s) != g or g >= 48:
                continue
            expanded += 1
            actions = [1, 2, 3, 4]
            if s[3][0] >= 0 or NEI[s[0]][s[1]] in s[2][self.na:]:
                actions.append(5)
            for action in actions:
                t = self.step(s, action)
                ng = g + 1
                if t == s or ng >= costs.get(t, INF):
                    continue
                costs[t], parents[t] = ng, (s, action)
                if sum(a >= 0 for a in t[2][self.na:]) < living:
                    return self.reconstruct(parents, t, state)
                h = self.enemy_h(t)
                if h < best_h:
                    best, best_h = t, h
                serial += 1
                heapq.heappush(heap, (ng + 2.2 * h, ng, serial, t))
        if best_h < h0:
            return self.reconstruct(parents, best, state)
        return None

    def navigation(self, start, blocked):
        parent = {start: None}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for action, q in enumerate(NEI[p], 1):
                if (q >= 0 and q not in parent and q not in blocked
                        and q not in self.hazards):
                    parent[q] = p, action
                    queue.append(q)
        return parent

    @staticmethod
    def route(parent, end):
        actions = []
        while parent[end] is not None:
            end, action = parent[end]
            actions.append(action)
        return actions[::-1]

    def transfer_costs(self, state, b):
        result = [INF] * 256
        fixed = self.fixed_boxes(state, b)
        for j in range(self.na):
            a = state[2][j]
            if a < 0:
                continue
            walking = self.walk_dist(a, fixed)
            busy = 0
            held = state[3][j + 1]
            if held >= 0 and held != b:
                q = state[4][held]
                offset = q % 16 - a % 16, q // 16 - a // 16
                duration = self.shape_dist(
                    offset, self.fixed_boxes(state, held)
                )[a]
                if duration >= INF:
                    continue
                busy = min(35, duration + 2)

            for q in range(256):
                if q in self.walls or q in fixed:
                    continue
                for a2 in NEI[q]:
                    if a2 < 0 or walking[a2] >= INF:
                        continue
                    offset = q % 16 - a2 % 16, q // 16 - a2 // 16
                    carrying = self.shape_dist(offset, fixed)[a2]
                    if carrying < INF:
                        result[q] = min(
                            result[q], walking[a2] + carrying + busy + 2
                        )
        return result

    def idle_action(self, state):
        p, rotation, actors, holds, boxes = state
        front = NEI[p][rotation]
        if (holds[0] < 0 and front not in boxes
                and front not in {a for a in actors[self.na:] if a >= 0}):
            return 5

        occupied = self.occupied(p, actors, boxes)
        for k, q in enumerate(NEI[p]):
            if k != rotation and (
                    q < 0 or q in occupied or q in self.hazards):
                return k + 1

        choices = []
        for k, q in enumerate(NEI[p]):
            if q < 0 or q in occupied or q in self.hazards:
                continue
            separation = min(
                (abs(q % 16 - a % 16) + abs(q // 16 - a // 16)
                 for a in actors if a >= 0),
                default=20
            )
            choices.append((-separation, k + 1))
        return min(choices)[1] if choices else 1

    def check_handoff(self, state, b, deadline, complete=False):
        s = state
        if self.delivered(s, b):
            return 0
        if not complete and self.viable_carrier(s, b):
            return 1
        unchanged = 0
        horizon = 48 if complete else 24
        for tick in range(1, horizon + 1):
            if time.monotonic() >= deadline:
                return None
            t = self.step(s, self.idle_action(s))
            if self.delivered(t, b):
                return tick
            if not complete and self.viable_carrier(t, b):
                return tick
            if t[2:] == s[2:]:
                unchanged += 1
            else:
                unchanged = 0
            s = t
            if unchanged >= 4:
                break
        return None

    def retreat(self, state, box, region):
        """Clear the inner side of a handoff before waiting for a carrier."""
        p = state[0]
        q = state[4][box]
        options = []
        for k, a in enumerate(NEI[p]):
            if a < 0 or a not in region:
                continue
            t = self.step(state, k + 1)
            if t[0] != a or t[3][0] >= 0:
                continue
            separation = abs(a % 16 - q % 16) + abs(a // 16 - q // 16)
            clearance = sum(v in region for v in NEI[a])
            options.append((-separation, -clearance, k + 1, t))
        if options:
            _, _, action, t = min(options, key=lambda v: v[:3])
            return [action], t
        return [], state

    def extraction_plan(self, state, region, deadline):
        """Push boxes through a player-blocking perimeter to outside carriers."""
        p, rotation, actors, holds, boxes = state
        occupied = self.occupied(p, actors, boxes)
        nav = self.navigation(p, occupied - {p})

        if holds[0] >= 0:
            tasks = [holds[0]]
        else:
            tasks = [
                b for b, q in enumerate(boxes)
                if b not in self.protected
                and not self.delivered(state, b)
                and (q in region or any(a in region for a in NEI[q]))
                and not self.viable_carrier(state, b)
            ]

        candidates = []
        for b in tasks:
            if time.monotonic() >= deadline:
                break
            q = boxes[b]
            transfer = self.transfer_costs(state, b)
            already = holds[0] == b
            starts = [p] if already else [
                a for a in NEI[q] if a >= 0 and a in nav
            ]

            for start in starts:
                pickup = [] if already else self.route(nav, start)
                facing = next((k for k, a in enumerate(NEI[start]) if a == q), None)
                if facing is None:
                    continue
                if not already:
                    current_rotation = pickup[-1] - 1 if pickup else rotation
                    if current_rotation != facing:
                        pickup.append(facing + 1)
                    pickup.append(5)

                offset = q % 16 - start % 16, q // 16 - start // 16
                blocked = occupied - {p, q}
                parent, distance = {start: None}, {start: 0}
                queue = deque([start])
                options = []
                while queue:
                    a = queue.popleft()
                    bq = self.shifted(a, offset)
                    if bq not in region and transfer[bq] < INF:
                        if bq != q or already:
                            length = len(pickup) + distance[a] + 2
                            score = length + 0.22 * transfer[bq]
                            options.append((score, a, bq))

                    for action, a2 in enumerate(NEI[a], 1):
                        if (a2 < 0 or a2 in parent or a2 in blocked
                                or a2 in self.hazards):
                            continue
                        b2 = self.shifted(a2, offset)
                        if b2 < 0 or b2 in blocked:
                            continue
                        parent[a2] = a, action
                        distance[a2] = distance[a] + 1
                        queue.append(a2)

                options.sort()
                for estimate, end, endpoint in options[:12]:
                    script = pickup + self.route(parent, end) + [5]
                    candidates.append(
                        (estimate, b, endpoint, script, len(pickup))
                    )

        candidates.sort(key=lambda v: v[0])
        best = None
        tested = set()
        fallback = None
        for estimate, b, endpoint, script, pickup_length in candidates[:100]:
            if time.monotonic() >= deadline:
                break
            if best is not None and estimate > best[0] + 7:
                break

            s, used = state, []
            valid = True
            for index, action in enumerate(script):
                s = self.step(s, action)
                used.append(action)
                if self.won(s):
                    return used
                if (pickup_length and index + 1 == pickup_length
                        and s[3][0] != b):
                    valid = False
                    break
            if not valid or s[3][0] >= 0:
                continue
            if s[4][b] in region:
                continue

            extra, s = self.retreat(s, b, region)
            used += extra
            if (b, s) in tested:
                continue
            tested.add((b, s))

            owner = next((i for i, k in enumerate(s[3]) if k == b), -1)
            if owner > self.na:
                continue
            if fallback is None and (owner < 0 or self.viable_carrier(s, b)):
                fallback = used

            ticks = self.check_handoff(s, b, deadline, complete=True)
            if ticks is None:
                continue
            rank = len(used) + 0.22 * ticks
            delivered_others = sum(
                self.delivered(s, k) and not self.delivered(state, k)
                for k in range(len(boxes)) if k != b
            )
            rank -= 1.5 * delivered_others
            if best is None or rank < best[0]:
                best = rank, used

        if best:
            return best[1]
        return fallback

    def box_plan(self, state, deadline):
        p, rotation, actors, holds, boxes = state
        if holds[0] >= 0 and boxes[holds[0]] in self.goals:
            return [5]

        occupied = self.occupied(p, actors, boxes)
        nav = self.navigation(p, occupied - {p})
        if holds[0] >= 0:
            tasks = [holds[0]]
        else:
            tasks = [
                b for b, q in enumerate(boxes)
                if b not in self.protected
                and q not in self.goals and not self.viable_carrier(state, b)
            ]

        candidates = []
        for b in tasks:
            if time.monotonic() >= deadline:
                break
            q = boxes[b]
            already = holds[0] == b
            transfer = self.transfer_costs(state, b) if self.na else [INF] * 256
            starts = [p] if already else [
                a for a in NEI[q] if a >= 0 and a in nav
            ]
            for start in starts:
                pickup = [] if already else self.route(nav, start)
                facing = next((k for k, v in enumerate(NEI[start]) if v == q), None)
                if facing is None:
                    continue
                if not already:
                    current_rotation = pickup[-1] - 1 if pickup else rotation
                    if current_rotation != facing:
                        pickup.append(facing + 1)
                    pickup.append(5)

                offset = q % 16 - start % 16, q // 16 - start // 16
                blocked = occupied - {p, q}
                parent, distance = {start: None}, {start: 0}
                queue = deque([start])
                direct_options, relay_options = [], []
                while queue:
                    a = queue.popleft()
                    bq = self.shifted(a, offset)
                    length = len(pickup) + distance[a] + 1
                    if bq in self.goals:
                        direct_options.append((float(length), a, bq))
                    elif transfer[bq] < INF:
                        immediate_helper = any(
                            adj(a2, bq) and state[3][j + 1] < 0
                            for j, a2 in enumerate(actors[:self.na])
                            if a2 >= 0
                        )
                        if bq != q or (already and immediate_helper):
                            relay_options.append((
                                length + 2 + 0.38 * transfer[bq], a, bq
                            ))

                    for action, a2 in enumerate(NEI[a], 1):
                        if (a2 < 0 or a2 in parent or a2 in blocked
                                or a2 in self.hazards):
                            continue
                        b2 = self.shifted(a2, offset)
                        if b2 < 0 or b2 in blocked:
                            continue
                        parent[a2] = a, action
                        distance[a2] = distance[a] + 1
                        queue.append(a2)

                direct_options.sort()
                relay_options.sort()
                for direct, options in (
                        (True, direct_options[:5]),
                        (False, relay_options[:5])):
                    for estimate, end, endpoint in options:
                        script = pickup + self.route(parent, end) + [5]
                        candidates.append((
                            estimate, b, direct, endpoint, script, len(pickup)
                        ))

        candidates.sort(key=lambda v: v[0])
        best = None
        tested_relays = set()
        checks = 0

        for estimate, b, direct, endpoint, script, pickup_length in candidates:
            if time.monotonic() >= deadline:
                break
            if best is not None and estimate > best[0] + 6:
                break
            s, used = state, []
            valid = True
            for index, action in enumerate(script):
                s = self.step(s, action)
                used.append(action)
                if self.won(s):
                    return used
                if (pickup_length and index + 1 == pickup_length
                        and s[3][0] != b):
                    valid = False
                    break
            if not valid:
                continue

            done = self.delivered(s, b)
            owner = next((i for i, k in enumerate(s[3]) if k == b), -1)
            if direct and not done:
                continue
            if not direct and not done:
                if not (0 < owner <= self.na or
                        (owner < 0 and s[4][b] == endpoint)):
                    continue
                if (b, s) in tested_relays:
                    continue
                tested_relays.add((b, s))
                checks += 1
                wait = self.check_handoff(s, b, deadline)
                if wait is None:
                    continue
                rank = estimate + 0.12 * wait
            else:
                rank = float(len(used))

            before = sum(self.delivered(state, k) for k in range(len(boxes)))
            after = sum(self.delivered(s, k) for k in range(len(boxes)))
            rank -= 2 * max(0, after - before - int(done))
            if best is None or rank < best[0]:
                best = rank, used
            if checks >= 24 and best is not None:
                break
        return best[1] if best else None

    def dynamic_plan(self, state, deadline):
        targets = [
            b for b in range(len(state[4]))
            if b not in self.protected
            and not self.delivered(state, b)
            and not self.viable_carrier(state, b)
        ]
        if state[3][0] >= 0:
            targets = [state[3][0]]
        if not targets:
            return None

        maps = {}
        for b in targets:
            if time.monotonic() >= deadline:
                return None
            fixed = self.fixed_boxes(state, b)
            transfer = self.transfer_costs(state, b) if self.na else [INF] * 256
            blocked = self.walls | set(fixed)

            for offset in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                valid = [False] * 256
                d = [float(INF)] * 256
                heap = []
                for a in range(256):
                    q = self.shifted(a, offset)
                    valid[a] = (
                        a not in blocked and a not in self.hazards
                        and q >= 0 and q not in blocked
                    )
                    if not valid[a]:
                        continue
                    cost = 0 if q in self.goals else (
                        2 + 0.15 * transfer[q] if transfer[q] < INF else INF
                    )
                    if cost < INF:
                        d[a] = cost
                        heapq.heappush(heap, (cost, a))
                while heap:
                    cost, a = heapq.heappop(heap)
                    if cost != d[a]:
                        continue
                    for a2 in NEI[a]:
                        if a2 >= 0 and valid[a2] and cost + 1 < d[a2]:
                            d[a2] = cost + 1
                            heapq.heappush(heap, (cost + 1, a2))
                maps[b, offset] = d

        def heuristic(s):
            p, _, actors, holds, boxes = s
            walk = self.walk_dist(p)
            value = float(INF)
            for b in targets:
                q = boxes[b]
                if self.delivered(s, b):
                    return 0.0
                if holds[0] == b:
                    off = q % 16 - p % 16, q // 16 - p // 16
                    table = maps.get((b, off))
                    if table is not None:
                        value = min(value, table[p] + 1)
                else:
                    for a in NEI[q]:
                        if a < 0:
                            continue
                        off = q % 16 - a % 16, q // 16 - a // 16
                        value = min(value, walk[a] + 2 + maps[b, off][a])
                    owner = next((i for i, k in enumerate(holds) if k == b), -1)
                    if 0 < owner <= self.na:
                        value = min(value, 2.0)
                    elif owner < 0:
                        for j, a in enumerate(actors[:self.na]):
                            if a < 0 or holds[j + 1] >= 0:
                                continue
                            wd = self.walk_dist(a)
                            for a2 in NEI[q]:
                                if a2 < 0:
                                    continue
                                off = q % 16 - a2 % 16, q // 16 - a2 // 16
                                if self.shape_dist(
                                        off, self.fixed_boxes(s, b))[a2] < INF:
                                    value = min(value, wd[a2] + 2.0)
            return value

        h0 = heuristic(state)
        if h0 >= INF:
            return None
        heap = [(2.0 * h0, 0, 0, state)]
        costs, parents = {state: 0}, {}
        serial, expanded = 0, 0
        best, best_h = state, h0

        while heap and expanded < 1600 and time.monotonic() < deadline:
            _, g, _, s = heapq.heappop(heap)
            if costs.get(s) != g or g >= 38:
                continue
            expanded += 1
            for action in (1, 2, 3, 4, 5):
                t = self.step(s, action)
                ng = g + 1
                if t == s or ng >= costs.get(t, INF):
                    continue
                costs[t], parents[t] = ng, (s, action)
                success = False
                for b in targets:
                    if self.delivered(t, b):
                        success = True
                        break
                    owner = next((i for i, k in enumerate(t[3]) if k == b), -1)
                    if 0 < owner <= self.na and self.viable_carrier(t, b):
                        success = True
                        break
                if success:
                    return self.reconstruct(parents, t, state)

                h = heuristic(t)
                if h + 0.08 * ng < best_h:
                    best, best_h = t, h + 0.08 * ng
                serial += 1
                heapq.heappush(heap, (ng + 2.0 * h, ng, serial, t))

        if best != state and best_h + 0.5 < h0:
            return self.reconstruct(parents, best, state)[:5]
        return None

    def waiting(self, state):
        scored = []
        for action in (1, 2, 3, 4, 5):
            t = self.step(state, action)
            if self.won(t):
                return action
            score = 0.0
            for b, q in enumerate(t[4]):
                if self.delivered(t, b):
                    continue
                score += 35
                owner = next((i for i, k in enumerate(t[3]) if k == b), -1)
                if 0 < owner <= self.na:
                    a = t[2][owner - 1]
                    off = q % 16 - a % 16, q // 16 - a // 16
                    score += min(
                        120, self.shape_dist(off, self.fixed_boxes(t, b))[a]
                    )
                else:
                    score += 20
            for a in t[2][:self.na]:
                if a >= 0:
                    distance = (
                        abs(t[0] % 16 - a % 16)
                        + abs(t[0] // 16 - a // 16)
                    )
                    score += 0.3 / (1 + distance)
            if t[3][0] >= 0 and state[3][0] < 0:
                score += 18
            if t[0] != state[0]:
                score += 0.005
            scored.append((score, action))
        return min(scored)[1]


def act(obs, available, memory):
    started = time.monotonic()
    state = read_board(np.asarray(obs), memory)
    na = memory["na"]
    memory["ne"] = len(state[2]) - na
    memory["tick"] = memory.get("tick", 0) + 1
    tick = memory["tick"]

    if len(memory.get("priority", ())) != len(state[4]):
        memory["priority"] = list(range(len(state[4])))

    previous = memory.get("last_state")
    if previous and len(previous[3]) == len(state[3]):
        priority = list(memory["priority"])
        for i in range(1, len(state[3])):
            b = state[3][i]
            if b < 0 or previous[3][i] == b:
                continue
            a = previous[2][i - 1]
            competitors = [
                k for k, q in enumerate(previous[4])
                if k != b and adj(a, q)
            ]
            if competitors and b in priority:
                priority.remove(b)
                index = min(
                    (priority.index(k) for k in competitors if k in priority),
                    default=len(priority)
                )
                priority.insert(index, b)
        memory["priority"] = priority

    model = Model(memory["env"], na, memory)
    expected = memory.get("expected")

    if (expected is not None and expected != state and previous is not None
            and na <= 4 and memory["ne"] <= 3):
        deadline = min(started + 0.65, time.monotonic() + 0.4)
        found = False
        for ao in itertools.permutations(range(na)):
            if found or time.monotonic() >= deadline:
                break
            for eo in itertools.permutations(range(memory["ne"])):
                model.order_a, model.order_e = ao, eo
                model.cache.clear()
                if model.step(previous, memory["last_action"]) == state:
                    memory["order_a"] = list(ao)
                    memory["order_e"] = list(eo)
                    found = True
                    break
                if time.monotonic() >= deadline:
                    break
        model = Model(memory["env"], na, memory)

    allowed = {
        int(name[6:]) for name in available
        if name.startswith("ACTION") and name[6:].isdigit()
    }

    region = model.component(state[0])
    enclosed = (
        na > 0 and bool(model.hazards)
        and 1 < len(region) < 100
        and not (region & model.goals)
        and any(a >= 0 and a not in region for a in state[2][:na])
    )

    # Protect outgoing shipments until they have had time to be collected.
    if enclosed:
        shipping = memory.setdefault("shipping", {})
        for b, q in enumerate(state[4]):
            owner = next((i for i, k in enumerate(state[3]) if k == b), -1)
            if model.delivered(state, b) or owner == 0 or q in region:
                shipping.pop(b, None)
                continue

            carrier_position = state[2][owner - 1] if owner > 0 else -1
            signature = q, owner, carrier_position
            old = shipping.get(b)
            if old is None or old[0] != signature:
                shipping[b] = (signature, tick)
            last_change = shipping[b][1]

            # Outside shipments are not repeatedly grabbed merely because
            # their carrier has not arrived yet.
            grace = 14 if owner > 0 else 22
            if tick - last_change < grace or q not in model.hazards:
                model.protected.add(b)

    history = memory.get("recent", [])
    signature = state[0], state[2], state[3], state[4]
    repetitions = history.count(signature)
    history.append(signature)
    memory["recent"] = history[-14:]

    if previous and state[2:] == previous[2:]:
        memory["stalled"] = memory.get("stalled", 0) + 1
    else:
        memory["stalled"] = 0

    plan = memory.get("plan", [])
    if expected != state:
        plan = []

    if not plan:
        if any(a >= 0 for a in state[2][na:]):
            plan = model.intercept(
                state, min(started + 3.4, time.monotonic() + 3.0)
            )

        if not plan and enclosed:
            plan = model.extraction_plan(state, region, started + 7.3)

            if not plan:
                unfinished_inside = any(
                    q in region and not model.delivered(state, b)
                    for b, q in enumerate(state[4])
                )
                if unfinished_inside and time.monotonic() < started + 7.5:
                    plan = model.dynamic_plan(state, started + 8.4)
                if not plan:
                    plan = [model.idle_action(state)]

        if not plan:
            force_search = repetitions >= 2 or memory["stalled"] >= 4
            if force_search:
                plan = model.dynamic_plan(
                    state, min(started + 5.0, time.monotonic() + 3.5)
                )
            if not plan:
                plan = model.box_plan(
                    state, min(started + 7.7, time.monotonic() + 3.3)
                )
            if not plan and time.monotonic() < started + 7.8:
                plan = model.dynamic_plan(state, started + 8.5)
            if not plan:
                plan = [model.waiting(state)]

    action = plan[0]
    if action not in allowed:
        action = next((a for a in (5, 1, 2, 3, 4) if a in allowed), None)
        plan = [action] if action is not None else []
    if action is None:
        return "RESET", None, None

    memory["plan"] = list(plan[1:])
    memory["last_state"] = state
    memory["last_action"] = action
    memory["expected"] = model.step(state, action)
    return "ACTION" + str(action), None, None

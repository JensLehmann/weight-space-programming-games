import numpy as np
from collections import deque
import heapq
import time

INF = 100000
DXY = ((0, -1), (0, 1), (-1, 0), (1, 0))
OFFSETS = (-16, 16, -1, 1)


def distance(a, b):
    return abs(a % 16 - b % 16) + abs(a // 16 - b // 16)


def adjacent(a, b):
    return a >= 0 and b >= 0 and distance(a, b) == 1


def front(p, direction):
    x, y = p % 16, p // 16
    dx, dy = DXY[direction]
    x, y = x + dx, y + dy
    return x + 16*y if 0 <= x < 16 and 0 <= y < 16 else -1


def read_board(obs):
    boxes, helpers, enemies = {}, [], []
    player, direction = None, 0
    walls, hazards, floors = set(), set(), {}
    occupied = set()

    for y in range(16):
        for x in range(16):
            q = x + 16*y
            a = obs[4*y:min(4*y+4, 63), 4*x:4*x+4]
            if not a.size:
                continue
            vals = set(int(v) for v in a.ravel())

            if 14 in vals:
                player = q
                occupied.add(q)
                scores = [
                    np.mean(a[0, :] == 0),
                    np.mean(a[-1, :] == 0) if a.shape[0] == 4 else -1,
                    np.mean(a[:, 0] == 0),
                    np.mean(a[:, -1] == 0),
                ]
                direction = int(np.argmax(scores))
                if max(scores) <= 0:
                    direction = 1
            elif np.count_nonzero(a == 12) >= 6:
                helpers.append(q)
                occupied.add(q)
            elif a.shape[0] >= 3 and np.all(a[1:3, 1:3] == 15):
                enemies.append(q)
                occupied.add(q)
            elif (a.shape[0] >= 3
                  and np.all(a[1:3, 1:3] == 9)
                  and int(a[0, 0]) in (0, 3, 4, 5)):
                boxes[q] = int(a[0, 0])
                occupied.add(q)
            elif vals == {5}:
                walls.add(q)
            elif vals <= {2, 9}:
                floors[q] = a.copy()
            elif (vals <= {1, 2} and 2 in vals and 1 in vals
                  and np.count_nonzero(a == 2) >= 6):
                hazards.add(q)
            elif vals != {1} and 2 not in vals and 9 not in vals:
                walls.add(q)

    goals, storage = set(), set()
    unseen = set(floors)
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = [start]
        for q in queue:
            x, y = q % 16, q // 16
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x+dx, y+dy
                n = nx + 16*ny
                if 0 <= nx < 16 and 0 <= ny < 16 and n in unseen:
                    unseen.remove(n)
                    component.add(n)
                    queue.append(n)

        is_goal = any(np.any(floors[q] == 9) for q in component)
        extended = set(component)
        if is_goal:
            for q in component:
                a = floors[q]
                x, y = q % 16, q // 16
                for dx, dy, edge in (
                    (-1, 0, a[:, 0]), (1, 0, a[:, -1]),
                    (0, -1, a[0, :]), (0, 1, a[-1, :])
                ):
                    nx, ny = x+dx, y+dy
                    n = nx + 16*ny
                    if (0 <= nx < 16 and 0 <= ny < 16
                            and n in occupied and np.any(edge == 2)):
                        extended.add(n)

        xs = [q % 16 for q in extended]
        ys = [q // 16 for q in extended]
        region = set(extended)
        for yy in range(min(ys), max(ys)+1):
            for xx in range(min(xs), max(xs)+1):
                q = xx + 16*yy
                if q in occupied:
                    region.add(q)
        (goals if is_goal else storage).update(region)

    return dict(p=player, d=direction, boxes=boxes, helpers=helpers,
                enemies=enemies, walls=walls, hazards=hazards,
                goals=goals, storage=storage)


def align(positions, predicted, previous=None):
    remaining = set(positions)
    if predicted is None:
        return tuple(sorted(remaining))

    result = [-1]*len(predicted)
    for i, q in enumerate(predicted):
        if q >= 0 and q in remaining:
            result[i] = q
            remaining.remove(q)

    if previous is not None:
        for i, q in enumerate(previous):
            if i < len(result) and result[i] < 0 and q in remaining:
                result[i] = q
                remaining.remove(q)

    while remaining:
        choices = []
        for i in range(len(result)):
            if result[i] >= 0:
                continue
            reference = predicted[i]
            if reference < 0 and previous is not None:
                reference = previous[i]
            for q in remaining:
                cost = distance(reference, q) if reference >= 0 else 100
                choices.append((cost, i, q))
        if not choices:
            result.extend(sorted(remaining))
            break
        _, i, q = min(choices)
        result[i] = q
        remaining.remove(q)
    return tuple(result)


class World:
    def __init__(self, board):
        self.walls = set(board["walls"])
        self.hazards = set(board["hazards"])
        self.goals = set(board["goals"])
        self.storage = set(board["storage"])
        self.neighbors = []
        for q in range(256):
            x, y = q % 16, q // 16
            ns = []
            # Autonomous agents use this ordering to break BFS ties.
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                xx, yy = x+dx, y+dy
                if 0 <= xx < 16 and 0 <= yy < 16:
                    ns.append(xx + 16*yy)
            self.neighbors.append(tuple(ns))
        self.rebuild()

    def update(self, board):
        before = (frozenset(self.walls), frozenset(self.hazards),
                  frozenset(self.goals), frozenset(self.storage))
        self.walls.update(board["walls"])
        self.hazards.update(board["hazards"])
        self.goals.update(board["goals"])
        self.storage.update(board["storage"])
        self.storage.difference_update(self.goals)
        after = (frozenset(self.walls), frozenset(self.hazards),
                 frozenset(self.goals), frozenset(self.storage))
        changed = before != after
        if changed:
            self.rebuild()
        self.helper_cache = {}
        return changed

    def rebuild(self):
        self.actor_blocked = self.walls | self.hazards
        self.distances = []
        for source in range(256):
            ds = [INF]*256
            if source not in self.actor_blocked:
                ds[source] = 0
                queue = deque([source])
                while queue:
                    q = queue.popleft()
                    for n in self.neighbors[q]:
                        if n not in self.actor_blocked and ds[n] == INF:
                            ds[n] = ds[q]+1
                            queue.append(n)
            self.distances.append(ds)
        self.field_cache = {}
        self.manip_cache = {}
        self.helper_cache = {}

    def pair_field(self, offset, destinations, fixed):
        wall = self.walls | set(fixed)
        actor_wall = wall | self.hazards
        valid = bytearray(256)
        for p in range(256):
            b = p+offset
            if (0 <= b < 256 and adjacent(p, b)
                    and p not in actor_wall and b not in wall):
                valid[p] = 1

        ds = [INF]*256
        queue = deque()
        for b in destinations:
            p = b-offset
            if 0 <= p < 256 and valid[p]:
                ds[p] = 0
                queue.append(p)
        while queue:
            p = queue.popleft()
            for n in self.neighbors[p]:
                if valid[n] and ds[n] == INF:
                    ds[n] = ds[p]+1
                    queue.append(n)
        return ds

    def manipulation_fields(self, destinations, fixed):
        """Relaxed distances permitting release and re-gripping."""
        key = frozenset(destinations), fixed
        if key in self.manip_cache:
            return self.manip_cache[key]

        wall = self.walls | set(fixed)
        actor_wall = wall | self.hazards
        valid = [bytearray(256) for _ in OFFSETS]
        ds = [[INF]*256 for _ in OFFSETS]
        queue = []

        for k, offset in enumerate(OFFSETS):
            for p in range(256):
                b = p+offset
                if (0 <= b < 256 and adjacent(p, b)
                        and p not in actor_wall and b not in wall):
                    valid[k][p] = 1
            for b in destinations:
                p = b-offset
                if 0 <= p < 256 and valid[k][p]:
                    ds[k][p] = 0
                    heapq.heappush(queue, (0, k, p))

        while queue:
            cost, k, p = heapq.heappop(queue)
            if ds[k][p] != cost:
                continue

            for n in self.neighbors[p]:
                if valid[k][n] and cost+1 < ds[k][n]:
                    ds[k][n] = cost+1
                    heapq.heappush(queue, (cost+1, k, n))

            b = p+OFFSETS[k]
            for j, offset in enumerate(OFFSETS):
                if j == k:
                    continue
                other = b-offset
                if not 0 <= other < 256 or not valid[j][other]:
                    continue
                walk = self.distances[p][other]
                if walk >= INF:
                    continue
                if OFFSETS[k] == -offset:
                    walk = max(4, walk)
                nc = cost+walk+2
                if nc < ds[j][other]:
                    ds[j][other] = nc
                    heapq.heappush(queue, (nc, j, other))

        result = {offset: ds[k] for k, offset in enumerate(OFFSETS)}
        if len(self.manip_cache) > 100:
            self.manip_cache.clear()
        self.manip_cache[key] = result
        return result

    def fields(self, state, target):
        boxes, helpers, owners = state[2], state[3], state[5]
        fixed = frozenset(
            b for i, b in enumerate(boxes)
            if i != target and b in self.goals and owners[i] == -1
        )
        key = fixed, tuple(helpers)
        if key in self.field_cache:
            return self.field_cache[key]

        destinations = self.goals-set(fixed)
        goal_fields = {
            o: self.pair_field(o, destinations, fixed) for o in OFFSETS
        }

        reachable = set()
        queue = deque()
        blocked = self.actor_blocked | set(fixed)
        for q in helpers:
            if q >= 0 and q not in blocked and q not in reachable:
                reachable.add(q)
                queue.append(q)
        while queue:
            q = queue.popleft()
            for n in self.neighbors[q]:
                if n not in blocked and n not in reachable:
                    reachable.add(n)
                    queue.append(n)

        handoff = set(destinations)
        for stand in reachable:
            for offset in OFFSETS:
                b = stand+offset
                if (0 <= b < 256 and adjacent(stand, b)
                        and goal_fields[offset][stand] < INF):
                    handoff.add(b)

        hand_fields = {
            o: self.pair_field(o, handoff, fixed) for o in OFFSETS
        }
        result = goal_fields, hand_fields, handoff, fixed
        if len(self.field_cache) > 80:
            self.field_cache.clear()
        self.field_cache[key] = result
        return result

    def path_step(self, start, occupancy, destinations, box=None):
        if box is None:
            if start in destinations:
                return start, 0
            offset = 0
            actor_occ = occupancy
            cargo_occ = None
        else:
            offset = box-start
            if box in destinations:
                return start, 0
            actor_occ = occupancy-{box}
            cargo_occ = occupancy-{start}

        seen = bytearray(256)
        seen[start] = 1
        first = [-1]*256
        depth = [0]*256
        queue = deque([start])

        while queue:
            q = queue.popleft()
            for n in self.neighbors[q]:
                if seen[n] or n in actor_occ or n in self.hazards:
                    continue
                if box is not None:
                    b = n+offset
                    if (not 0 <= b < 256 or not adjacent(n, b)
                            or b in cargo_occ):
                        continue
                    finished = b in destinations
                else:
                    finished = n in destinations

                seen[n] = 1
                first[n] = n if q == start else first[q]
                depth[n] = depth[q]+1
                if finished:
                    return first[n], depth[n]
                queue.append(n)
        return None, INF

    def helper_path(self, state, target):
        key = state, target
        if key in self.helper_cache:
            return self.helper_cache[key]
        p, d, boxes, helpers, enemies, owners = state
        owner = owners[target]
        if not 1 <= owner <= len(helpers):
            return None, INF

        occupancy = self.walls | {p} | set(boxes) | set(helpers)
        occupancy.update(e for e in enemies if e >= 0)
        result = self.path_step(
            helpers[owner-1], occupancy, self.goals, boxes[target]
        )
        if len(self.helper_cache) > 16000:
            self.helper_cache.clear()
        self.helper_cache[key] = result
        return result

    def step(self, state, action):
        p, d, bt, ht, et, ot = state
        boxes, helpers, enemies, owners = (
            list(bt), list(ht), list(et), list(ot)
        )
        nh = len(helpers)
        occupancy = self.walls | {p} | set(boxes) | set(helpers)
        occupancy.update(e for e in enemies if e >= 0)

        def carried(actor):
            try:
                return owners.index(actor)
            except ValueError:
                return -1

        def move(pos, ci, dest):
            if dest < 0 or dest in self.hazards:
                return pos
            if ci < 0:
                if dest in occupancy:
                    return pos
                occupancy.remove(pos)
                occupancy.add(dest)
                return dest

            old_box = boxes[ci]
            new_box = dest+old_box-pos
            if not 0 <= new_box < 256 or not adjacent(dest, new_box):
                return pos
            if dest in occupancy and dest != old_box:
                return pos
            if new_box in occupancy and new_box != pos:
                return pos

            occupancy.remove(pos)
            occupancy.remove(old_box)
            occupancy.add(dest)
            occupancy.add(new_box)
            boxes[ci] = new_box
            return dest

        held = carried(0)
        if 1 <= action <= 4:
            if held < 0:
                d = action-1
            p = move(p, held, front(p, action-1))
        elif action == 5:
            if held >= 0:
                owners[held] = -1
            else:
                f = front(p, d)
                for i, b in enumerate(boxes):
                    if b == f:
                        owners[i] = 0
                        break
                for i, e in enumerate(enemies):
                    if e >= 0 and e == f:
                        ci = carried(1+nh+i)
                        if ci >= 0:
                            owners[ci] = -1
                        occupancy.remove(e)
                        enemies[i] = -1
                        break

        # Friendly collectors.
        for j in range(nh):
            actor, pos = j+1, helpers[j]
            ci = carried(actor)
            if ci >= 0:
                if boxes[ci] in self.goals:
                    owners[ci] = -1
                else:
                    nxt, length = self.path_step(
                        pos, occupancy, self.goals, boxes[ci]
                    )
                    if nxt is not None and length:
                        helpers[j] = move(pos, ci, nxt)
            else:
                pickup = -1
                for i, b in enumerate(boxes):
                    if (owners[i] == -1 and b not in self.goals
                            and adjacent(pos, b)):
                        pickup = i
                        break
                if pickup >= 0:
                    owners[pickup] = actor
                    break
                targets = set()
                for i, b in enumerate(boxes):
                    if owners[i] == -1 and b not in self.goals:
                        targets.update(self.neighbors[b])
                if targets:
                    nxt, length = self.path_step(pos, occupancy, targets)
                    if nxt is not None and length:
                        helpers[j] = move(pos, -1, nxt)

        # Thieves can steal from both the player and friendly collectors.
        for j, pos in enumerate(enemies):
            if pos < 0:
                continue
            actor = 1+nh+j
            ci = carried(actor)
            if ci >= 0:
                if boxes[ci] in self.storage:
                    owners[ci] = -1
                else:
                    nxt, length = self.path_step(
                        pos, occupancy, self.storage, boxes[ci]
                    )
                    if nxt is not None and length:
                        enemies[j] = move(pos, ci, nxt)
            else:
                pickup = -1
                for i, b in enumerate(boxes):
                    if (owners[i] <= nh and b not in self.storage
                            and adjacent(pos, b)):
                        pickup = i
                        break
                if pickup >= 0:
                    owners[pickup] = actor
                    break
                targets = set()
                for i, b in enumerate(boxes):
                    if owners[i] <= nh and b not in self.storage:
                        targets.update(self.neighbors[b])
                if targets:
                    nxt, length = self.path_step(pos, occupancy, targets)
                    if nxt is not None and length:
                        enemies[j] = move(pos, -1, nxt)

        return p, d, tuple(boxes), tuple(helpers), tuple(enemies), tuple(owners)

    def complete(self, state):
        return all(b in self.goals and o == -1
                   for b, o in zip(state[2], state[5]))


def observe_state(board, prediction, previous, world):
    if prediction is None:
        boxes = tuple(sorted(board["boxes"]))
        helpers = tuple(sorted(board["helpers"]))
        enemies = tuple(sorted(board["enemies"]))
    else:
        boxes = align(board["boxes"], prediction[2],
                      previous[2] if previous else None)
        helpers = align(board["helpers"], prediction[3],
                        previous[3] if previous else None)
        enemies = align(board["enemies"], prediction[4],
                        previous[4] if previous else None)

    p, d = board["p"], board["d"]
    nh = len(helpers)
    actors = (p,)+helpers+enemies
    owners = [-1]*len(boxes)
    used = set()

    for i, b in enumerate(boxes):
        if board["boxes"].get(b) == 0 and adjacent(p, b):
            owners[i] = 0
            used.add(0)

    ambiguous = []
    for i, b in enumerate(boxes):
        if owners[i] >= 0:
            continue
        border = board["boxes"].get(b, 4)
        if border not in (3, 5):
            continue

        candidates = []
        for actor in range(1, len(actors)):
            pos = actors[actor]
            if pos < 0 or not adjacent(pos, b):
                continue
            score = 0
            if prediction is not None and i < len(prediction[5]):
                if prediction[5][i] == actor:
                    score += 20
            if previous is not None and i < len(previous[5]):
                if previous[5][i] == actor:
                    old_actors = (previous[0],)+previous[3]+previous[4]
                    if actor < len(old_actors):
                        if b-pos == previous[2][i]-old_actors[actor]:
                            score += 8
            if actor <= nh:
                score += 1
            candidates.append((score, actor))

        if candidates:
            if border == 5:
                ambiguous.append((i, candidates, True))
            elif prediction is not None and prediction[5][i] >= 1:
                ambiguous.append((i, candidates, False))
            elif prediction is None and b not in world.goals:
                ambiguous.append((i, candidates, False))

    ambiguous.sort(key=lambda item: (
        not item[2], len(item[1]), -max(v[0] for v in item[1])
    ))
    for i, candidates, mandatory in ambiguous:
        for score, actor in sorted(candidates, reverse=True):
            if actor not in used:
                owners[i] = actor
                used.add(actor)
                break

    return p, d, boxes, helpers, enemies, tuple(owners)


def reconstruct(parents, state):
    path = []
    while parents[state] is not None:
        prev, action = parents[state]
        path.append(action)
        state = prev
    path.reverse()
    return path


def search(world, initial, actions, heuristic, goal, deadline,
           max_nodes=7000, weight=1.7):
    if goal(initial):
        return [], True
    h0 = heuristic(initial)
    if h0 >= INF:
        return [], False

    serial = 0
    queue = [(weight*h0, 0, serial, 0, initial)]
    costs = {initial: 0}
    parents = {initial: None}
    best, best_h, best_g = initial, h0, 0
    expanded = 0

    while queue and expanded < max_nodes:
        if time.perf_counter() >= deadline:
            break
        _, _, _, cost, state = heapq.heappop(queue)
        if costs.get(state) != cost:
            continue
        expanded += 1

        if world.complete(state) or goal(state):
            return reconstruct(parents, state), True

        for action in actions:
            nxt = world.step(state, action)
            if nxt == state:
                continue
            nc = cost+1
            if nc >= costs.get(nxt, INF):
                continue
            costs[nxt] = nc
            parents[nxt] = state, action
            if world.complete(nxt) or goal(nxt):
                return reconstruct(parents, nxt), True

            h = heuristic(nxt)
            if h >= INF:
                continue
            if h < best_h or (h == best_h and nc < best_g):
                best, best_h, best_g = nxt, h, nc
            serial += 1
            delivered = sum(
                b in world.goals and o == -1
                for b, o in zip(nxt[2], nxt[5])
            )
            heapq.heappush(queue, (
                nc+weight*h, -delivered, serial, nc, nxt
            ))

    if best != initial and best_h+1 < h0:
        return reconstruct(parents, best)[:1], False
    return [], False


def box_heuristic(world, state, target, fields, collector):
    p, d, boxes, helpers, enemies, owners = state
    b, owner = boxes[target], owners[target]

    if b in world.goals and owner == -1:
        return 0

    if 1 <= owner <= len(helpers):
        hp = helpers[owner-1]
        offset = b-hp
        if offset in collector and collector[offset][hp] < INF:
            step, length = world.helper_path(state, target)
            if step is not None:
                return 0.25

    if owner == 0:
        offset = b-p
        if offset in fields and fields[offset][p] < INF:
            return fields[offset][p]+1

    extra = 1 if 0 in owners else 0
    best = INF
    for stand in world.neighbors[b]:
        if stand in world.actor_blocked:
            continue
        offset = b-stand
        travel = world.distances[p][stand]
        carry = fields[offset][stand]
        if travel < INF and carry < INF:
            best = min(best, travel+1+carry+1+extra)
    return best


def wait_action(world, state, actions, visits):
    best_action, best_score = actions[0], -1e30
    goals = tuple(world.goals)
    for action in actions:
        nxt = world.step(state, action)
        delivered = sum(
            b in world.goals and o == -1
            for b, o in zip(nxt[2], nxt[5])
        )
        score = 10000*delivered
        score -= 4000*sum(e >= 0 for e in nxt[4])
        if 0 in nxt[5]:
            score -= 300
        if nxt[0] == state[0]:
            score += 2
        if goals:
            for b, owner in zip(nxt[2], nxt[5]):
                if b not in world.goals:
                    score -= min(distance(b, g) for g in goals)
                if 1 <= owner <= len(nxt[3]):
                    score += 0.5
        score += 0.03*sum(
            n not in world.actor_blocked for n in world.neighbors[nxt[0]]
        )
        score += 0.002*sum(distance(nxt[0], h) for h in nxt[3])
        # Preserve stationary waiting when collectors are making progress;
        # penalize it only if the full board state has already repeated.
        score -= 3*max(0, min(8, visits.get(nxt, 0))-1)
        if action == 5:
            score += 0.001
        if score > best_score:
            best_score, best_action = score, action
    return best_action


def act(obs, available, memory):
    deadline = time.perf_counter()+8.2
    available_set = set(available)
    actions = [
        i for i in (1, 2, 3, 4, 5)
        if "ACTION%d" % i in available_set
    ]
    if not actions:
        name = "RESET" if "RESET" in available_set else available[0]
        return name, None, None

    board = read_board(np.asarray(obs))
    if board["p"] is None:
        name = "RESET" if "RESET" in available_set else "ACTION%d" % actions[0]
        return name, None, None

    world = memory.get("_world")
    if world is None:
        world = World(board)
        memory["_world"] = world
        changed = True
    else:
        changed = world.update(board)

    previous = memory.get("_state")
    prediction = memory.get("_prediction")
    state = observe_state(board, prediction, previous, world)

    visits = memory.setdefault("_visits", {})
    visits[state] = visits.get(state, 0)+1
    if len(visits) > 600:
        visits.clear()
        visits[state] = 1

    plan = memory.get("_plan", [])
    if changed or prediction != state:
        plan = []

    def emit(action, remaining=()):
        memory["_state"] = state
        memory["_prediction"] = world.step(state, action)
        memory["_plan"] = list(remaining)
        return "ACTION%d" % action, None, None

    if plan and plan[0] in actions:
        return emit(plan[0], plan[1:])

    p, d, boxes, helpers, enemies, owners = state
    alive = sum(e >= 0 for e in enemies)

    # Remove thieves before transporting boxes into their reach.
    if alive:
        def enemy_goal(s):
            return sum(e >= 0 for e in s[4]) < alive

        def enemy_h(s):
            pp, dd = s[0], s[1]
            best = INF
            for e in s[4]:
                if e < 0:
                    continue
                for stand in world.neighbors[e]:
                    travel = world.distances[pp][stand]
                    if travel >= INF:
                        continue
                    turn = 0.7 if pp == stand and front(pp, dd) != e else 0
                    best = min(best, travel+1+turn)
            if 0 in s[5]:
                best += 1
            return best

        path, solved = search(
            world, state, actions, enemy_h, enemy_goal,
            deadline, max_nodes=9000, weight=1.9
        )
        if path:
            return emit(path[0], path[1:] if solved else [])
        return emit(wait_action(world, state, actions, visits))

    candidates = []
    recovery_candidates = []
    held = owners.index(0) if 0 in owners else -1

    for target, b in enumerate(boxes):
        if time.perf_counter() >= deadline-0.4:
            break
        owner = owners[target]
        if b in world.goals and owner == -1:
            continue

        collector, hand_fields, handoff, fixed = world.fields(state, target)

        if 1 <= owner <= len(helpers):
            step, length = world.helper_path(state, target)
            if step is not None:
                continue

        direct_cost = box_heuristic(
            world, state, target, collector, collector
        )
        mode, fields, cost = "goal", collector, direct_cost

        # Do not keep dropping and re-grabbing a box that has already
        # been left in a viable collection area.
        if helpers and (b not in handoff or owner == 0):
            hand_cost = box_heuristic(
                world, state, target, hand_fields, collector
            )
            if hand_cost+2 < direct_cost:
                mode, fields, cost = "handoff", hand_fields, hand_cost

        priority_bonus = 1000 if target == held else 0
        if cost < INF:
            candidates.append(
                (cost-priority_bonus, target, mode, fields,
                 collector, handoff, False)
            )
        else:
            recovery_candidates.append(
                (target, collector, handoff, fixed, priority_bonus)
            )

    # Re-gripping is a fallback, not a cheaper substitute for every
    # ordinary delivery. That keeps normal transport plans decisive.
    for target, collector, handoff, fixed, bonus in recovery_candidates:
        if time.perf_counter() >= deadline-0.4:
            break
        destinations = world.goals-set(fixed)
        direct = world.manipulation_fields(destinations, fixed)
        direct_cost = box_heuristic(
            world, state, target, direct, collector
        )
        mode, fields, cost = "goal", direct, direct_cost

        if helpers:
            hand = world.manipulation_fields(handoff, fixed)
            hand_cost = box_heuristic(
                world, state, target, hand, collector
            )
            if hand_cost+2 < direct_cost:
                mode, fields, cost = "handoff", hand, hand_cost

        if cost < INF:
            candidates.append(
                (cost-bonus+2, target, mode, fields,
                 collector, handoff, True)
            )

    candidates.sort(key=lambda item: item[0])
    selected = candidates[:3]

    for rank, candidate in enumerate(selected):
        if time.perf_counter() >= deadline-0.1:
            break
        _, target, mode, fields, collector, handoff, recovery = candidate
        initial_box = boxes[target]
        initially_held = owners[target] == 0

        def box_goal(s, target=target, mode=mode, handoff=handoff,
                     initial_box=initial_box,
                     initially_held=initially_held, recovery=recovery):
            b, owner = s[2][target], s[5][target]
            if b in world.goals and owner == -1:
                return True
            if 1 <= owner <= len(s[3]):
                step, length = world.helper_path(s, target)
                if step is not None:
                    return True

            if (not recovery and mode == "handoff"
                    and owner == -1 and b in handoff):
                return b != initial_box or initially_held
            return False

        def box_h(s, target=target, fields=fields, collector=collector):
            return box_heuristic(world, s, target, fields, collector)

        now = time.perf_counter()
        remaining = deadline-now
        local_deadline = min(
            deadline,
            now+(remaining if rank == len(selected)-1
                 else max(0.4, remaining*0.65))
        )
        path, solved = search(
            world, state, actions, box_h, box_goal,
            local_deadline, max_nodes=8500 if recovery else 6500,
            weight=1.7
        )
        if path:
            return emit(path[0], path[1:] if solved else [])

    return emit(wait_action(world, state, actions, visits))

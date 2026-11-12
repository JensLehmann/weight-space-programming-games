import time
import heapq
from collections import deque
import numpy as np


# Actions: up, down, left, right.
_DXY = ((0, -1), (0, 1), (-1, 0), (1, 0))
_OPP = (1, 0, 3, 2)


def _xy(p):
    return 4 + 6 * (p % 10), 4 + 6 * (p // 10)


def _neighbor(p, a):
    x, y = p % 10, p // 10
    dx, dy = _DXY[a]
    x, y = x + dx, y + dy
    return x + 10 * y if 0 <= x < 10 and 0 <= y < 10 else -1


def _patch(obs, p, radius):
    x, y = _xy(p)
    if x < radius or y < radius or x + radius >= 64 or y + radius >= 64:
        return None
    return obs[y-radius:y+radius+1, x-radius:x+radius+1]


def _actors(obs, color):
    out = []
    for p in range(10, 100):
        q = _patch(obs, p, 2)
        if q is not None and q[2, 2] == 5:
            if np.count_nonzero(q == color) == 24:
                out.append(p)
    return out


def _gate_template(color, purple=False):
    a = np.full((7, 7), 5, dtype=np.int8)
    a[1:6, 1:6] = color
    if not purple:
        a[1, 2] = 5
        a[1, 4] = 5
    a[6, 3] = color
    return a


def _goal_templates():
    a = np.full((9, 9), -1, dtype=np.int8)
    a[1, 1:8] = 9
    a[7, 1:8] = 9
    a[1:8, 7] = 9
    a[2:7, 2:7] = 5
    a[4, 4] = 9
    return [np.rot90(a, -k) for k in range(4)]


def _components(obs, color):
    labels = np.full((64, 64), -1, dtype=np.int16)
    n = 0
    for y, x in zip(*np.where(obs == color)):
        y, x = int(y), int(x)
        if labels[y, x] >= 0:
            continue
        labels[y, x] = n
        q = [(x, y)]
        for xx, yy in q:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = xx + dx, yy + dy
                if 0 <= nx < 64 and 0 <= ny < 64:
                    if obs[ny, nx] == color and labels[ny, nx] < 0:
                        labels[ny, nx] = n
                        q.append((nx, ny))
        n += 1
    return labels


class _World:
    def __init__(self, obs):
        self.walk = set()
        self.gates = []       # (closed cell, open cell, color, rotation)
        self.buttons = []     # (cell, color, component key)
        self.portals = []
        self.goal = -1
        self.start = -1

        gate_variants = []
        for color in (8, 11, 15):
            for purple in (False, True):
                t = _gate_template(color, purple)
                for k in range(4):
                    gate_variants.append((color, k, np.rot90(t, -k)))

        goal_variants = _goal_templates()
        for p in range(10, 100):
            q = _patch(obs, p, 2)
            if q is not None and np.count_nonzero(q) >= 22:
                self.walk.add(p)

            r = _patch(obs, p, 3)
            if r is not None:
                for color, k, template in gate_variants:
                    if np.array_equal(r, template):
                        # Rotation 0 opens down; 90 left; 180 up; 270 right.
                        direction = (1, 2, 0, 3)[k]
                        self.gates.append((p, _neighbor(p, direction), color, k))
                        self.walk.add(p)
                        break

                if (np.all(r[0, :] == 15) and
                    np.all(r[6, :] == 15) and
                    np.all(r[:, 0] == 15) and
                    np.all(r[:, 6] == 15)):
                    self.portals.append(p)
                    self.walk.add(p)

            r = _patch(obs, p, 4)
            if r is not None:
                for template in goal_variants:
                    mask = template >= 0
                    if np.array_equal(r[mask], template[mask]):
                        self.goal = p
                        self.walk.add(p)
                        break

        agents = _actors(obs, 9)
        if agents:
            self.start = agents[0]
            self.walk.add(self.start)

        gate_cells = {g[0] for g in self.gates}
        labels = {c: _components(obs, c) for c in (8, 11, 15)}
        for p in range(10, 100):
            if p in gate_cells:
                continue
            q = _patch(obs, p, 1)
            if q is None:
                continue
            for c in (8, 11, 15):
                if np.all(q == c):
                    x, y = _xy(p)
                    key = (c, int(labels[c][y, x]))
                    self.buttons.append((p, c, key))
                    self.walk.add(p)
                    break

        keys = []
        for _, _, key in self.buttons:
            if key not in keys:
                keys.append(key)
        self.group_at = {}
        self.group_regular = [0] * len(keys)
        self.group_toggle = [0] * len(keys)
        self.group_portals = [[] for _ in keys]

        for p, c, key in self.buttons:
            self.group_at[p] = keys.index(key)

        for i, (p, opened, c, rotation) in enumerate(self.gates):
            x, y = _xy(p)
            key = (c, int(labels[c][y, x]))
            if key in keys:
                group = keys.index(key)
                if c == 11:
                    self.group_toggle[group] |= 1 << i
                else:
                    self.group_regular[group] |= 1 << i

        for p in self.portals:
            x, y = _xy(p)
            key = (15, int(labels[15][y-3, x]))
            if key in keys:
                self.group_portals[keys.index(key)].append(p)

        self.targets = []
        for p, c, key in self.buttons:
            group = self.group_at[p]
            useful = (self.group_regular[group] or
                      len(self.group_portals[group]) >= 2)
            if useful:
                self.targets.append((p, c, group))
        # Portal occupants can occasionally be useful recordings too.
        for p in self.portals:
            self.targets.append((p, 0, -1))

        enemies = _actors(obs, 14)
        self.has_enemy = bool(enemies)
        self.enemy_start = enemies[0] if enemies else -1
        if self.has_enemy:
            self.walk.add(self.enemy_start)

        slots = 0
        for i in range(12):
            x = 1 + 4 * i
            if x + 3 > 64:
                break
            q = obs[1:4, x:x+3]
            if np.count_nonzero(q) >= 8:
                slots += 1
            else:
                break
        self.capacity = max(0, min(3, slots - 1))

        self.neighbors = {
            p: tuple(_neighbor(p, a) if _neighbor(p, a) in self.walk else -1
                     for a in range(4))
            for p in self.walk
        }
        positions = (self.start,)
        if self.has_enemy:
            positions += (self.enemy_start,)
        self.initial = (positions, 0, 0, 0, 1)

    def blocked(self, mask):
        return {
            g[1] if mask & (1 << i) else g[0]
            for i, g in enumerate(self.gates)
        }

    def occupied_groups(self, positions):
        return {self.group_at[p] for p in positions if p in self.group_at}

    def signals(self, mask, falling, rising):
        for group in falling:
            mask &= ~self.group_regular[group]
        for group in rising:
            mask |= self.group_regular[group]
            mask ^= self.group_toggle[group]
        return mask

    def swap(self, positions, groups):
        positions = list(positions)
        for group in groups:
            portals = self.group_portals[group]
            if len(portals) >= 2:
                a, b = portals[0], portals[1]
                positions = [b if p == a else a if p == b else p
                             for p in positions]
        return positions

    def closed_kills(self, old_mask, new_mask, positions, alive):
        closing = old_mask & ~new_mask
        for i, g in enumerate(self.gates):
            if closing & (1 << i):
                if positions[0] == g[0]:
                    return True, alive
                if self.has_enemy and positions[-1] == g[0]:
                    alive = 0
        return False, alive

    def step(self, state, action, scripts, reverse_index=None,
             enemy_reverse=None, departure_undo=False, arrival_undo=False):
        positions, mask, phase, heading, alive = state
        old = positions
        new = list(old)
        moving = set()
        blocked = self.blocked(mask)

        def destination(p, a):
            if p not in self.neighbors:
                return -1
            d = self.neighbors[p][a]
            return d if d not in blocked else -1

        d = destination(old[0], action)
        if d < 0 and reverse_index is None:
            return None, None
        if d >= 0:
            new[0] = d
            moving.add(0)

        index = phase if reverse_index is None else reverse_index
        for j, script in enumerate(scripts):
            if index >= len(script):
                continue
            a = script[index]
            if reverse_index is not None:
                a = _OPP[a]
            actor = j + 1
            d = destination(old[actor], a)
            if d >= 0:
                new[actor] = d
                moving.add(actor)

        enemy_move = None
        if self.has_enemy:
            e = len(old) - 1
            if reverse_index is not None:
                if enemy_reverse is not None:
                    d = destination(old[e], enemy_reverse)
                    if d >= 0:
                        new[e] = d
                        moving.add(e)
            elif alive:
                # Heading numbering: down, right, up, left.
                heading_actions = (1, 3, 0, 2)
                order = (heading, (heading-1) % 4,
                         (heading+1) % 4, (heading+2) % 4)
                for h in order:
                    a = heading_actions[h]
                    d = destination(old[e], a)
                    if d >= 0:
                        heading = h
                        enemy_move = a
                        new[e] = d
                        moving.add(e)
                        break

        before = self.occupied_groups(old)
        middle = self.occupied_groups(
            tuple(p for i, p in enumerate(old) if i not in moving)
        )
        falling = before - middle
        mid_mask = self.signals(mask, falling, set())
        dead, alive = self.closed_kills(mask, mid_mask, new, alive)
        if dead:
            return None, enemy_move

        if departure_undo:
            new = self.swap(new, falling)

        after = self.occupied_groups(new)
        rising = after - middle
        final_mask = self.signals(mid_mask, set(), rising)
        dead, alive = self.closed_kills(mid_mask, final_mask, new, alive)
        if dead:
            return None, enemy_move

        if not arrival_undo:
            new = self.swap(new, rising)

        # Teleported occupants are registered on their destination pads.
        again = self.occupied_groups(new)
        if again != after:
            updated = self.signals(final_mask, after - again, again - after)
            dead, alive = self.closed_kills(final_mask, updated, new, alive)
            if dead:
                return None, enemy_move
            final_mask = updated

        if self.has_enemy and alive and new[0] == new[-1]:
            return None, enemy_move

        cap = max((len(s) for s in scripts), default=0)
        if reverse_index is None:
            next_phase = min(phase + 1, cap)
        else:
            next_phase = min(reverse_index, cap)
        return (tuple(new), final_mask, next_phase, heading, alive), enemy_move

    def rewind(self, state, path, enemy_history, scripts):
        n = len(path)
        for i in range(n-1, -1, -1):
            enemy_reverse = None
            if i < len(enemy_history) and enemy_history[i] is not None:
                enemy_reverse = _OPP[enemy_history[i]]
            state, _ = self.step(
                state, _OPP[path[i]], scripts,
                reverse_index=i,
                enemy_reverse=enemy_reverse,
                departure_undo=(i < n-1),
                arrival_undo=(i > 0)
            )
            if state is None:
                return None

        positions, mask, phase, heading, alive = state
        positions = list(positions)
        positions[0] = self.start
        if self.has_enemy:
            positions.insert(len(positions)-1, self.start)
        else:
            positions.append(self.start)
        return (tuple(positions), mask, 0, heading, alive)

    def record(self, root, path, scripts):
        state = root
        history = []
        for a in path:
            state, e = self.step(state, a, scripts)
            if state is None:
                return None
            history.append(e)
        return self.rewind(state, path, history, scripts)


def _reconstruct(parent, state):
    path = []
    while parent[state] is not None:
        previous, action = parent[state]
        path.append(action)
        state = previous
    path.reverse()
    return path


def _search(world, root, scripts, deadline, collect=True, limit=45000):
    if root[0][0] == world.goal:
        return [], []
    parent = {root: None}
    queue = deque([root])
    candidates = []
    seen_targets = set()
    target_map = {p: (c, g) for p, c, g in world.targets}
    toggle_mask = 0
    for m in world.group_toggle:
        toggle_mask |= m
    endpoint_set = set()
    for script in scripts:
        state = world.initial
        p = world.start
        # Only a duplicate-endpoint heuristic, not part of simulation.
        for a in script:
            q = _neighbor(p, a)
            if q >= 0:
                p = q
        endpoint_set.add(p)

    count = 0
    while queue:
        if count % 128 == 0 and time.monotonic() >= deadline:
            break
        state = queue.popleft()
        count += 1
        p = state[0][0]
        if p == world.goal:
            return _reconstruct(parent, state), candidates

        if collect and p in target_map and parent[state] is not None:
            color, group = target_map[p]
            key = (p, state[1] & toggle_mask)
            if key not in seen_targets and p not in endpoint_set:
                seen_targets.add(key)
                path = _reconstruct(parent, state)
                if len(path) <= 65:
                    candidates.append((path, color, group, p))
                if len(candidates) >= 18:
                    collect = False

        for a in range(4):
            child, _ = world.step(state, a, scripts)
            if child is not None and child not in parent:
                parent[child] = (state, a)
                queue.append(child)
        if len(parent) >= limit:
            break
    return None, candidates


def _plan(world, root, scripts, deadline):
    heap = []
    serial = 0
    heapq.heappush(heap, (0, serial, root, tuple(scripts), [], 0))
    fallback = None
    expanded = set()

    while heap and time.monotonic() < deadline:
        priority, _, state, old_scripts, prefix, penalty = heapq.heappop(heap)
        key = (state, old_scripts)
        if key in expanded:
            continue
        expanded.add(key)

        can_record = len(old_scripts) < world.capacity
        goal_path, candidates = _search(
            world, state, old_scripts, deadline, collect=can_record
        )
        if goal_path is not None:
            return prefix + goal_path
        if not can_record:
            continue

        # Prefer pressure switches over recordings on teleport pads.
        candidates.sort(key=lambda c: (c[1] == 0, len(c[0])))
        for path, color, group, endpoint in candidates:
            variants = [path]
            if color == 15 and group >= 0:
                if len(world.group_portals[group]) >= 2 and path:
                    # Repeated visits provide later teleport activation times.
                    last = path[-1]
                    for k in (1, 2, 3, 5, 7, 10):
                        variants.append(path + [_OPP[last], last] * k)

            for recording in variants:
                if time.monotonic() >= deadline:
                    break
                child = world.record(state, recording, old_scripts)
                if child is None:
                    continue
                new_scripts = old_scripts + (tuple(recording),)
                actions = prefix + recording + [4]  # 4 denotes ACTION5.
                extra_penalty = 12 if color == 0 else 0
                new_penalty = penalty + extra_penalty
                serial += 1
                heapq.heappush(
                    heap, (len(actions) + new_penalty, serial,
                           child, new_scripts, actions, new_penalty)
                )
                if fallback is None:
                    fallback = actions
    return fallback


def _observe_state(world, obs, predicted, previous_enemy):
    positions, mask, phase, heading, alive = predicted
    positions = list(positions)
    players = _actors(obs, 9)
    if players:
        positions[0] = players[0]

    ghosts = _actors(obs, 2)
    end = len(positions) - (1 if world.has_enemy else 0)
    if ghosts:
        for i in range(1, end):
            x, y = _xy(positions[i])
            positions[i] = min(
                ghosts,
                key=lambda p: abs(_xy(p)[0]-x) + abs(_xy(p)[1]-y)
            )

    if world.has_enemy:
        enemies = _actors(obs, 14)
        if enemies:
            e = enemies[0]
            if previous_enemy is not None and previous_enemy >= 0:
                dx = e % 10 - previous_enemy % 10
                dy = e // 10 - previous_enemy // 10
                if abs(dx) + abs(dy) == 1:
                    heading = {(0, 1): 0, (1, 0): 1,
                               (0, -1): 2, (-1, 0): 3}[(dx, dy)]
            positions[-1] = e
            alive = 1
        else:
            alive = 0

    for i, (closed, opened, color, rotation) in enumerate(world.gates):
        template = np.rot90(_gate_template(color, color == 11), -rotation)
        q = _patch(obs, closed, 3)
        r = _patch(obs, opened, 3) if opened >= 0 else None
        if q is not None and np.array_equal(q, template):
            mask &= ~(1 << i)
        elif r is not None and np.array_equal(r, template):
            mask |= 1 << i

    return (tuple(positions), mask, phase, heading, alive)


def _emit(action, available):
    name = "ACTION" + str(action + 1)
    if name not in available:
        name = next((a for a in available if a != "RESET"), available[0])
    return name, None, None


def act(obs, available, memory):
    obs = np.asarray(obs)
    started = time.monotonic()
    deadline = started + 8.5

    # Make RESET a level reset even when the episode begins with action_count=0.
    # An invalid direction is a harmless action and does not advance ghosts.
    if not memory:
        memory["boot"] = 1
        if "RESET" in available:
            rough = _World(obs)
            if rough.start >= 0:
                blocked = rough.blocked(0)
                for a in range(4):
                    p = _neighbor(rough.start, a)
                    if p not in rough.walk or p in blocked:
                        return _emit(a, available)
            return _emit(4, available)

    if memory.get("boot") == 1 and "RESET" in available:
        memory["boot"] = 2
        return "RESET", None, None

    if "world" not in memory:
        world = _World(obs)
        memory["world"] = world
        memory["state"] = world.initial
        memory["scripts"] = []
        memory["segment"] = []
        memory["enemy_history"] = []
        memory["plan"] = []
        memory["previous_enemy"] = world.enemy_start
        memory["boot"] = 2
        if world.start < 0 or world.goal < 0:
            return _emit(4, available)
        memory["plan"] = _plan(world, world.initial, [], deadline) or []

    world = memory["world"]
    scripts = memory["scripts"]
    predicted = memory["state"]
    state = _observe_state(
        world, obs, predicted, memory.get("previous_enemy")
    )
    mismatch = state[0][0] != predicted[0][0]
    memory["state"] = state
    if world.has_enemy:
        memory["previous_enemy"] = state[0][-1]

    if mismatch:
        direct, _ = _search(world, state, scripts, deadline, collect=False)
        memory["plan"] = direct or []

    if not memory["plan"]:
        if not memory["segment"]:
            memory["plan"] = _plan(world, state, scripts, deadline) or []
        else:
            direct, _ = _search(world, state, scripts, deadline, collect=False)
            if direct is not None:
                memory["plan"] = direct
            elif len(scripts) < world.capacity and state[0][0] in world.group_at:
                memory["plan"] = [4]

    if not memory["plan"]:
        if "RESET" in available:
            memory.clear()
            memory["boot"] = 2
            return "RESET", None, None
        for a in range(4):
            child, _ = world.step(state, a, scripts)
            if child is not None:
                memory["state"] = child
                memory["segment"].append(a)
                return _emit(a, available)
        return _emit(4, available)

    action = memory["plan"].pop(0)

    if action == 4:
        path = memory["segment"]
        if path:
            child = world.rewind(
                state, path, memory["enemy_history"], scripts
            )
            if child is not None and len(scripts) < world.capacity:
                scripts.append(tuple(path))
                memory["state"] = child
            else:
                memory["plan"] = []
            memory["segment"] = []
            memory["enemy_history"] = []
        return _emit(4, available)

    child, enemy_move = world.step(state, action, scripts)
    if child is None:
        # The observed patrol may differ from the inferred invisible track.
        direct, _ = _search(world, state, scripts, deadline, collect=False)
        if direct:
            action = direct[0]
            memory["plan"] = direct[1:]
            child, enemy_move = world.step(state, action, scripts)
        if child is None:
            if "RESET" in available:
                memory.clear()
                memory["boot"] = 2
                return "RESET", None, None
            return _emit(4, available)

    memory["state"] = child
    memory["segment"].append(action)
    memory["enemy_history"].append(enemy_move)
    return _emit(action, available)

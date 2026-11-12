import numpy as np
from collections import defaultdict, deque
import heapq


_ROBOT = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 0, 0, 1],
], dtype=bool)

_GOAL = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=bool)

_CHECKPOINT = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
], dtype=bool)

# Effects of the binary instructions.
_INSTRUCTIONS = {
    1: ("move", -4, 0),
    2: ("move", 4, 0),
    3: ("move", 0, 4),
    5: ("rotate", 1),
    6: ("rotate", -1),
    7: ("rotate", 2),
    8: ("scale", 1),
    9: ("scale", -1),
    10: ("move", 8, 0),
    11: ("move", 8, 0),
    12: ("move", -8, 0),
    13: ("move", -8, 0),
    14: ("color", 9),
    15: ("color", 8),
    16: ("rotate", -1),
    33: ("move", 0, -4),
    34: ("move", -4, 0),
    63: ("color", 15),
}


def _components(mask):
    """Four-connected components, returned as (x0,y0,x1,y1,pixels)."""
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        queue = [(x, y)]
        pixels = []
        lo_x = hi_x = x
        lo_y = hi_y = y
        while queue:
            xx, yy = queue.pop()
            pixels.append((xx, yy))
            lo_x = min(lo_x, xx)
            hi_x = max(hi_x, xx)
            lo_y = min(lo_y, yy)
            hi_y = max(hi_y, yy)
            for nx, ny in ((xx-1, yy), (xx+1, yy),
                           (xx, yy-1), (xx, yy+1)):
                if (0 <= nx < w and 0 <= ny < h and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    queue.append((nx, ny))
        result.append((lo_x, lo_y, hi_x+1, hi_y+1, pixels))
    return result


def _matches(obs, template, colors, scales, rotations=True, bottom=51):
    result = []
    for scale in scales:
        for rotation in range(4 if rotations else 1):
            mask = np.rot90(template, -rotation)
            mask = np.repeat(np.repeat(mask, scale, 0), scale, 1)
            h, w = mask.shape
            if h > bottom-2 or w > 64:
                continue
            ay, ax = np.argwhere(mask)[0]
            ay, ax = int(ay), int(ax)
            for color in colors:
                ys, xs = np.nonzero(obs[:bottom] == color)
                for py, px in zip(ys, xs):
                    x, y = int(px)-ax, int(py)-ay
                    if x < 0 or y < 2 or x+w > 64 or y+h > bottom:
                        continue
                    patch = obs[y:y+h, x:x+w]
                    if (np.all(patch[mask] == color) and
                            np.all(patch[~mask] != color)):
                        result.append((x, y, rotation, scale, color))
    return list(dict.fromkeys(result))


def _read_program(obs):
    # Horizontal bits are three pixels wide, surrounded by the black panel.
    groups = defaultdict(list)
    for y in range(30, 50):
        for x in range(1, 61):
            color = int(obs[y, x])
            if color not in (1, 5):
                continue
            if not np.all(obs[y, x:x+3] == color):
                continue
            if (obs[y, x-1] != 0 or obs[y, x+3] != 0 or
                    not np.all(obs[y-1, x:x+3] == 0) or
                    not np.all(obs[y+1, x:x+3] == 0)):
                continue
            # The following vertical bit must also have the expected shape.
            c = x+1
            vertical = obs[y+2:y+5, c]
            if len(vertical) != 3:
                continue
            if not (np.all(vertical == 1) or np.all(vertical == 5)):
                continue
            groups[y].append(c)

    candidates = []
    for y, cols in groups.items():
        cols = sorted(set(cols))
        if not cols:
            continue
        # Keep regularly spaced runs rather than unrelated decorations.
        runs = []
        run = [cols[0]]
        for c in cols[1:]:
            if c-run[-1] == 5:
                run.append(c)
            else:
                runs.append(run)
                run = [c]
        runs.append(run)
        for run in runs:
            if len(run) >= 2:
                candidates.append((sum(run)/len(run), len(run), y, run))

    if not candidates:
        return None
    _, _, first_y, cols = max(candidates)
    rows = sorted(y for y, cs in groups.items()
                  if sorted(set(cs)) == cols)
    # Only the contiguous rows of this program.
    rows = [y for y in rows if (y-first_y) % 6 == 0]
    if not rows:
        rows = [first_y]
    rows = sorted(rows)
    if len(rows) > 3:
        rows = rows[:3]

    bit_locations = []
    values = []
    for c in cols:
        locations = []
        value = 0
        for j, y in enumerate(rows):
            locations.append((y, c))
            locations.append((y+3, c))
            if int(obs[y, c]) == 5:
                value |= 1 << (2*j)
            if int(obs[y+3, c]) == 5:
                value |= 1 << (2*j+1)
        bit_locations.append(locations)
        values.append(value)
    return {
        "cols": cols,
        "rows": rows,
        "bits": len(rows)*2,
        "locations": bit_locations,
        "values": values,
    }


def _button(obs):
    mask = np.isin(obs, [9, 10])
    mask[:50] = False
    best = None
    for x0, y0, x1, y1, pixels in _components(mask):
        w, h = x1-x0, y1-y0
        if 7 <= w <= 11 and 7 <= h <= 11 and len(pixels) >= 55:
            candidate = (len(pixels), x0, y0, x1, y1)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return None
    _, x0, y0, x1, y1 = best
    return ((y0+y1-1)//2, (x0+x1-1)//2)


def _board(obs, goal, program):
    gx, gy = goal[:2]
    bottom = min(program["rows"])-1
    options = []
    for y in range(2, min(16, bottom)):
        x = 0
        while x < 64:
            if obs[y, x] != 0:
                x += 1
                continue
            start = x
            while x < 64 and obs[y, x] == 0:
                x += 1
            if x-start >= 20 and start <= gx < x:
                options.append((y, start, x))
    if options:
        y, left, right = min(options)
        return (left+1, y+1, right-1, bottom)
    # Conservative fallback based on the checkerboard.
    mask = obs[:bottom] == 4
    ys, xs = np.nonzero(mask)
    near = (xs >= max(0, gx-32))
    if np.any(near):
        return (int(xs[near].min()), int(ys[near].min()),
                int(xs[near].max())+1, bottom)
    return (0, 3, 64, bottom)


def _make_world(obs, robot, goal, program, old=None):
    board = _board(obs, goal, program)
    left, top, right, bottom = board

    checkpoints = []
    for x, y, _, s, _ in _matches(
            obs, _CHECKPOINT, [11], range(1, 5),
            rotations=False, bottom=bottom):
        if left-2 <= x < right+2 and top-2 <= y < bottom:
            checkpoints.append((x, y, s))

    signature = (goal, tuple(program["cols"]),
                 tuple(program["rows"]), board)
    if old is not None and old.get("signature") == signature:
        checkpoints += old["checkpoints"]
    checkpoints = sorted(set(checkpoints))

    blocked = np.zeros((64, 64), dtype=np.int16)

    # Register complete wall components, including their top-left positions.
    # A sprite whose top-left lies outside its arena is not a game obstacle.
    for x0, y0, x1, y1, pixels in _components(obs == 6):
        if (left-2 <= x0 < right+2 and top-2 <= y0 < bottom):
            for x, y in pixels:
                blocked[y, x] = 1

    # Unfamiliar arena objects are treated conservatively as hazards.
    unusual = np.zeros((64, 64), dtype=bool)
    x0, x1 = max(0, left), min(64, right)
    y0, y1 = max(2, top), min(64, bottom)
    patch = obs[y0:y1, x0:x1]
    unusual[y0:y1, x0:x1] = ~np.isin(patch, [4, 5, 6, 11])

    def erase_object(x, y, size):
        xa, xb = max(0, x), min(64, x+size)
        ya, yb = max(0, y), min(64, y+size)
        if xa < xb and ya < yb:
            unusual[ya:yb, xa:xb] = False

    erase_object(robot[0], robot[1], 4*robot[3])
    erase_object(goal[0]-goal[3], goal[1]-goal[3], 6*goal[3])
    for x, y, s in checkpoints:
        erase_object(x, y, 4*s)

    for xa, ya, xb, yb, pixels in _components(unusual):
        blocked[ya:yb, xa:xb] = 1

    integral = np.pad(blocked, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    max_scale = min(6, max([robot[3], goal[3]] +
                         [c[2] for c in checkpoints]) + 1)

    return {
        "signature": signature,
        "board": board,
        "checkpoints": checkpoints,
        "integral": integral,
        "max_scale": max_scale,
        "collision_cache": {},
    }


def _blocked(world, x, y, scale):
    key = (x, y, scale)
    cache = world["collision_cache"]
    if key in cache:
        return cache[key]
    size = 4*scale
    xa, xb = max(0, x), min(64, x+size)
    ya, yb = max(0, y), min(64, y+size)
    if xa >= xb or ya >= yb:
        answer = False
    else:
        p = world["integral"]
        answer = bool(p[yb, xb]-p[ya, xb]-p[yb, xa]+p[ya, xa])
    cache[key] = answer
    return answer


def _inside_search(world, x, y):
    left, top, right, bottom = world["board"]
    return left-16 <= x <= right+16 and top-16 <= y <= bottom+16


def _transition(world, state, effect):
    x, y, r, s, color = state
    kind = effect[0]
    if kind == "move":
        nx, ny = x+effect[1], y+effect[2]
        if not _inside_search(world, nx, ny):
            return None
        if not _blocked(world, nx, ny, s):
            x, y = nx, ny
    elif kind == "rotate":
        r = (r+effect[1]) % 4
    elif kind == "scale":
        delta = effect[1]
        ns = max(1, s+delta)
        if ns > world["max_scale"]:
            return None
        if _blocked(world, x, y, ns):
            # Mirrors the simulator's adjustment-and-reversal behavior.
            ns = max(1, ns-delta)
        s = ns
        if s > world["max_scale"]:
            return None
    elif kind == "color":
        color = effect[1]
    return (x, y, r, s, color)


def _on_checkpoint(world, state):
    x, y, _, s, _ = state
    for cx, cy, cs in world["checkpoints"]:
        if cs != s:
            continue
        if x < cx+4*cs and x+4*s > cx and y < cy+4*cs and y+4*s > cy:
            return True
    return False


def _effects(bits, start_color, goal_color):
    groups = defaultdict(list)
    for code in range(1 << bits):
        effect = _INSTRUCTIONS.get(code, ("noop",))
        # Intermediate color changes cannot improve movement or collisions.
        if effect[0] == "color" and effect[1] != goal_color:
            continue
        if effect[0] == "color" and start_color == goal_color:
            continue
        groups[effect].append(code)
    return list(groups.items())


def _relaxed_neighbors(world, state, movement_effects):
    x, y, s = state
    for effect in movement_effects:
        if effect[0] == "move":
            nx, ny, ns = x+effect[1], y+effect[2], s
        else:
            nx, ny, ns = x, y, max(1, s+effect[1])
            if ns > world["max_scale"]:
                continue
        if (nx, ny, ns) == state:
            continue
        if not _inside_search(world, nx, ny):
            continue
        if _blocked(world, nx, ny, ns):
            continue
        yield (nx, ny, ns)


def _checkpoint_estimates(world, robot, goal, n, effect_groups):
    """Minimum optimistic number of full runs from each checkpoint position."""
    rem_x, rem_y = robot[0] % 4, robot[1] % 4
    anchors = set()
    for cx, cy, s in world["checkpoints"]:
        # Every lattice position whose bounding box overlaps this checkpoint.
        for x in range(cx-4*s+1, cx+4*s):
            if x % 4 != rem_x:
                continue
            for y in range(cy-4*s+1, cy+4*s):
                if y % 4 == rem_y and not _blocked(world, x, y, s):
                    anchors.add((x, y, s))
    anchors = sorted(anchors)
    if not anchors:
        return {}

    destination = (goal[0], goal[1], goal[3])
    movements = [effect for effect, codes in effect_groups
                 if effect[0] in ("move", "scale")]
    movements = list(dict.fromkeys(movements))
    anchor_set = set(anchors)
    reverse = defaultdict(list)
    direct = set()

    for origin in anchors:
        distances = {origin: 0}
        queue = deque([origin])
        while queue:
            state = queue.popleft()
            d = distances[state]
            if state == destination:
                direct.add(origin)
            if state in anchor_set and state != origin:
                reverse[state].append(origin)
            if d >= n:
                continue
            for nxt in _relaxed_neighbors(world, state, movements):
                if nxt not in distances:
                    distances[nxt] = d+1
                    queue.append(nxt)

    runs = {}
    queue = deque()
    for origin in direct:
        runs[origin] = 1
        queue.append(origin)
    while queue:
        state = queue.popleft()
        for origin in reverse[state]:
            if origin not in runs:
                runs[origin] = runs[state]+1
                queue.append(origin)
    return runs


def _plan(world, robot, goal, program, visits):
    current = program["values"]
    n = len(current)
    groups = _effects(program["bits"], robot[4], goal[4])

    # For each effect and slot, use its cheapest equivalent binary encoding.
    slot_effects = []
    for value in current:
        choices = []
        for effect, codes in groups:
            code = min(codes, key=lambda c: ((c ^ value).bit_count(), c))
            choices.append((effect, (code ^ value).bit_count(), code))
        choices.sort(key=lambda item: item[1])
        slot_effects.append(choices)

    dp = {robot: (0, ())}
    transition_cache = {}
    for choices in slot_effects:
        following = {}
        for state, (cost, path) in dp.items():
            for effect, extra, code in choices:
                key = (state, effect)
                if key not in transition_cache:
                    transition_cache[key] = _transition(world, state, effect)
                nxt = transition_cache[key]
                if nxt is None:
                    continue
                new_cost = cost+extra
                old = following.get(nxt)
                if old is None or new_cost < old[0]:
                    following[nxt] = (new_cost, path+(code,))
        dp = following
        if not dp:
            return None
        # Safety cap for unusually large evaluation layouts.
        if len(dp) > 55000:
            ranked = sorted(
                dp.items(),
                key=lambda item: (
                    item[1][0] +
                    (abs(item[0][0]-goal[0]) +
                     abs(item[0][1]-goal[1]))/12
                )
            )
            dp = dict(ranked[:55000])

    if goal in dp:
        return list(dp[goal][1])

    terminals = [(state, info) for state, info in dp.items()
                 if state != robot and _on_checkpoint(world, state)]
    if not terminals:
        return None

    estimates = _checkpoint_estimates(world, robot, goal, n, groups)

    def rank(item):
        state, (cost, path) = item
        x, y, r, s, color = state
        future = estimates.get((x, y, s), 100)
        repeats = visits.get(state, 0)
        distance = (abs(x-goal[0])+abs(y-goal[1]))/4
        orientation = int(r != goal[2])
        scale_error = abs(s-goal[3])
        color_error = int(color != goal[4])
        if future < 100:
            score = cost+1 + future*(n+3) + repeats*12
        else:
            score = 1000 + distance*3 + cost + repeats*20
        return (score, future, distance + orientation + scale_error +
                color_error, cost)

    state, (cost, path) = min(terminals, key=rank)
    return list(path)


def _click_difference(program, desired):
    for i, (actual, wanted) in enumerate(zip(program["values"], desired)):
        difference = actual ^ wanted
        if difference:
            bit = (difference & -difference).bit_length()-1
            return program["locations"][i][bit]
    return None


def act(obs, available, memory):
    obs = np.asarray(obs)
    available = set(available)
    if "ACTION6" not in available:
        name = "RESET" if "RESET" in available else next(iter(available))
        return (name, None, None)

    program = _read_program(obs)
    button = _button(obs)
    if program is None or button is None:
        if "RESET" in available:
            memory.clear()
            return ("RESET", None, None)
        return ("ACTION6", 58, 57)

    program_key = (tuple(program["cols"]), tuple(program["rows"]))
    pending = memory.get("desired")
    if pending is not None and memory.get("program_key") == program_key:
        click = _click_difference(program, pending)
        if click is not None:
            return ("ACTION6", int(click[0]), int(click[1]))
        memory.pop("desired", None)
        memory["just_ran"] = True
        return ("ACTION6", int(button[0]), int(button[1]))

    bottom = min(program["rows"])-1
    goal_matches = _matches(
        obs, _GOAL, [11, 9, 8, 15], range(1, 5), bottom=bottom)
    robot_matches = _matches(
        obs, _ROBOT, [11, 9, 8, 15], range(1, 5), bottom=bottom)

    if not goal_matches or not robot_matches:
        memory.pop("desired", None)
        if "RESET" in available:
            memory.clear()
            return ("RESET", None, None)
        return ("ACTION6", int(button[0]), int(button[1]))

    # The controllable arena is the rightmost one; demonstration sprites use 4.
    goal_sprite = max(goal_matches, key=lambda a: (a[0], a[1]))
    gx, gy, gr, gs, gc = goal_sprite
    goal = (gx+gs, gy+gs, gr, gs, gc)

    old_robot = memory.get("robot")
    if old_robot is not None:
        robot = min(robot_matches, key=lambda a:
                    abs(a[0]-old_robot[0])+abs(a[1]-old_robot[1]))
    else:
        robot = max(robot_matches, key=lambda a: a[0])
    robot = tuple(robot)

    old_world = memory.get("world")
    world = _make_world(obs, robot, goal, program, old_world)
    if old_world is None or old_world.get("signature") != world["signature"]:
        memory["visits"] = {}
        memory["failures"] = 0

    memory["world"] = world
    memory["robot"] = robot
    visits = memory.setdefault("visits", {})
    if memory.pop("just_ran", False):
        visits[robot] = visits.get(robot, 0)+1

    desired = _plan(world, robot, goal, program, visits)
    if desired is None:
        # A run can change unfamiliar timed objects. Reobserve before resetting.
        failures = memory.get("failures", 0)+1
        memory["failures"] = failures
        if failures > 2 and "RESET" in available:
            memory.clear()
            return ("RESET", None, None)
        memory["just_ran"] = True
        return ("ACTION6", int(button[0]), int(button[1]))

    memory["failures"] = 0
    memory["desired"] = desired
    memory["program_key"] = program_key
    click = _click_difference(program, desired)
    if click is not None:
        return ("ACTION6", int(click[0]), int(click[1]))

    memory.pop("desired", None)
    memory["just_ran"] = True
    return ("ACTION6", int(button[0]), int(button[1]))

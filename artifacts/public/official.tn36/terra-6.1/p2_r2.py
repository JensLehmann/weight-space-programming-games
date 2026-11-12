import numpy as np

# Templates are represented as masks; negative sprite pixels are transparent.
_PLAYER = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 0, 0, 1],
], dtype=np.uint8)

_GOAL = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1],
], dtype=np.uint8)


def _templates(base, max_scale):
    result = []
    for scale in range(1, max_scale + 1):
        for rot in range(4):
            mask = np.rot90(base, -rot)
            mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
            result.append((scale, rot, mask.astype(bool)))
    return result


_PLAYER_TEMPLATES = _templates(_PLAYER, 4)
_GOAL_TEMPLATES = _templates(_GOAL, 3)


def _find_pattern(grid, templates, colors, xmin=27, ymax=42):
    """Find exact rendered masks in the active board area."""
    matches = []
    for color in colors:
        color_map = grid == color
        for scale, rot, mask in templates:
            h, w = mask.shape
            # A positive template pixel provides a cheap preliminary test.
            py, px = np.argwhere(mask)[0]
            for y in range(3, min(ymax, 64 - h) + 1):
                for x in range(xmin, 64 - w + 1):
                    if not color_map[y + py, x + px]:
                        continue
                    region = color_map[y:y + h, x:x + w]
                    if np.array_equal(region, mask):
                        matches.append((x, y, rot, scale, color))
    return matches


def _find_panel(g):
    """
    Find the black instruction panel. Each column consists of horizontal
    and vertical three-pixel strokes, colored 1 or 5.
    """
    horizontal = {}
    for y in range(30, 49):
        xs = []
        for x in range(1, 59):
            c = int(g[y + 1, x + 1])
            if c not in (1, 5):
                continue
            if not (g[y + 1, x + 2] == c and g[y + 1, x + 3] == c):
                continue
            if not (g[y + 1, x] == 0 and g[y + 1, x + 4] == 0):
                continue
            if not (g[y, x + 2] == 0 and g[y + 2, x + 2] == 0):
                continue
            vc = int(g[y + 3, x + 2])
            if vc not in (1, 5):
                continue
            if not (g[y + 4, x + 2] == vc and g[y + 5, x + 2] == vc):
                continue
            if not (g[y + 4, x + 1] == 0 and g[y + 4, x + 3] == 0):
                continue
            xs.append(x)
        if xs:
            horizontal[y] = xs

    best = None
    for y, xs in horizontal.items():
        if len(xs) < 2:
            continue
        # Slots are spaced five pixels apart. Consider each possible phase.
        for phase in range(5):
            group = sorted(x for x in xs if x % 5 == phase)
            if len(group) < 2:
                continue
            # Keep the largest consecutive sequence.
            runs = []
            run = []
            for x in group:
                if run and x != run[-1] + 5:
                    runs.append(run)
                    run = []
                run.append(x)
            if run:
                runs.append(run)
            for run in runs:
                if len(run) < 2:
                    continue
                rows = 1
                while rows < 3:
                    next_xs = horizontal.get(y + 6 * rows, [])
                    if not all(x in next_xs for x in run):
                        break
                    rows += 1
                score = (len(run) * rows, rows, len(run), -y)
                if best is None or score > best[0]:
                    best = (score, y, run, rows)

    if best is None:
        return None

    _, y, xs, rows = best
    values = []
    for x in xs:
        value = 0
        for r in range(rows):
            if g[y + 6 * r + 1, x + 2] == 5:
                value |= 1 << (2 * r)
            if g[y + 6 * r + 4, x + 2] == 5:
                value |= 1 << (2 * r + 1)
        values.append(value)
    return y, xs, rows * 2, values


def _find_button(g):
    # The run button is a filled, rounded 9-by-9 sprite near the bottom.
    best = None
    for y in range(53, 61):
        for x in range(30, 62):
            if y - 2 < 0 or y + 3 > 64 or x - 2 < 0 or x + 3 > 64:
                continue
            inner = int(np.count_nonzero(g[y - 2:y + 3, x - 2:x + 3] == 9))
            outer = int(np.count_nonzero(
                g[max(0, y - 4):min(64, y + 5),
                  max(0, x - 4):min(64, x + 5)] == 9
            ))
            score = (inner, outer)
            if best is None or score > best[0]:
                best = (score, y, x)
    if best is not None and best[0][0] >= 20:
        return best[1], best[2]
    return 58, 57


def _find_checkpoints(g, player):
    px, py, _, _, _ = player
    found = []
    for y in range(3, 39):
        if (y - py) % 4:
            continue
        for x in range(13, 58):
            if (x - px) % 4:
                continue
            region = g[y:y + 4, x:x + 4] == 11
            if region.shape != (4, 4):
                continue
            for parity in (0, 1):
                mask = np.fromfunction(
                    lambda yy, xx: (xx + yy) % 2 == parity,
                    (4, 4), dtype=int
                )
                if np.array_equal(region, mask):
                    found.append((x, y))
                    break
    return found


def _wall_map(g, player):
    # Walls are four-pixel-grid-aligned in the game. Filling a tile with
    # any visible wall pixels also recovers wall pixels hidden by sprites.
    px, py, _, _, _ = player
    blocked = np.zeros((64, 64), dtype=np.int32)
    ys, xs = np.nonzero(g == 6)
    for y, x in zip(ys.tolist(), xs.tolist()):
        x0 = px + ((x - px) // 4) * 4
        y0 = py + ((y - py) // 4) * 4
        xa, xb = max(0, x0), min(64, x0 + 4)
        ya, yb = max(0, y0), min(64, y0 + 4)
        if xa < xb and ya < yb:
            blocked[ya:yb, xa:xb] = 1
    integral = np.zeros((65, 65), dtype=np.int32)
    integral[1:, 1:] = blocked.cumsum(axis=0).cumsum(axis=1)
    return integral


def _operation(code):
    if code in (1, 34):
        return "left"
    if code == 2:
        return "right"
    if code == 3:
        return "down"
    if code == 33:
        return "up"
    if code == 5:
        return "cw"
    if code in (6, 16):
        return "ccw"
    if code == 7:
        return "half"
    if code == 8:
        return "grow"
    if code == 9:
        return "shrink"
    if code in (10, 11):
        return "right2"
    if code in (12, 13):
        return "left2"
    if code == 14:
        return "color9"
    if code == 15:
        return "color8"
    if code == 63:
        return "color15"
    return "noop"


def _search(player, goal, panel_values, bits, walls, checkpoints, visited):
    sx, sy, sr, ss, sc = player
    gx, gy, gr, gs, gc = goal
    start = (sx, sy, sr, ss, sc)
    target = (gx, gy, gr, gs, gc)

    max_scale = max(2, ss, gs)
    max_scale = min(max_scale, 4)
    xmin = max(-4, min(sx, gx) - 24)
    xmax = min(63, max(sx, gx) + 24)
    ymin = max(-4, min(sy, gy) - 20)
    ymax = min(49, max(sy, gy) + 20)

    def collides(x, y, scale):
        size = 4 * scale
        xa, xb = max(0, x), min(64, x + size)
        ya, yb = max(0, y), min(64, y + size)
        if xa >= xb or ya >= yb:
            return False
        return bool(
            int(walls[yb, xb]) - int(walls[ya, xb])
            - int(walls[yb, xa]) + int(walls[ya, xa])
        )

    def transition(state, op):
        x, y, rot, scale, color = state
        dx = dy = 0
        if op == "left":
            dx = -4
        elif op == "right":
            dx = 4
        elif op == "down":
            dy = 4
        elif op == "up":
            dy = -4
        elif op == "right2":
            dx = 8
        elif op == "left2":
            dx = -8
        elif op == "cw":
            rot = (rot + 1) % 4
        elif op == "ccw":
            rot = (rot - 1) % 4
        elif op == "half":
            rot = (rot + 2) % 4
        elif op == "grow":
            new_scale = scale + 1
            if new_scale <= max_scale and not collides(x, y, new_scale):
                scale = new_scale
        elif op == "shrink":
            new_scale = max(1, scale - 1)
            if not collides(x, y, new_scale):
                scale = new_scale
        elif op == "color9":
            color = 9
        elif op == "color8":
            color = 8
        elif op == "color15":
            color = 15

        if dx or dy:
            nx, ny = x + dx, y + dy
            if not (xmin <= nx <= xmax and ymin <= ny <= ymax):
                return None
            if not collides(nx, ny, scale):
                x, y = nx, ny
        return x, y, rot, scale, color

    # For each slot and operation, use the encoding requiring fewest clicks.
    choices = []
    for current in panel_values:
        by_op = {}
        for code in range(1 << bits):
            op = _operation(code)
            if op.startswith("color"):
                desired_color = int(op[5:])
                if desired_color != gc:
                    continue
            cost = (current ^ code).bit_count()
            old = by_op.get(op)
            if old is None or (cost, code) < old:
                by_op[op] = (cost, code)
        choices.append([(op, cost, code)
                        for op, (cost, code) in by_op.items()])

    # Dynamic programming over the fixed number of instruction slots.
    states = {start: (0, ())}
    for slot_choices in choices:
        next_states = {}
        for state, (cost, path) in states.items():
            for op, extra, code in slot_choices:
                new_state = transition(state, op)
                if new_state is None:
                    continue
                new_cost = cost + extra
                old = next_states.get(new_state)
                if old is None or new_cost < old[0]:
                    next_states[new_state] = (new_cost, path + (code,))
        states = next_states
        if not states:
            return None

    if target in states:
        return list(states[target][1])

    # When one run cannot finish, aim for an unvisited checkpoint.
    checkpoint_set = set(checkpoints)
    candidates = []
    for state, (cost, path) in states.items():
        x, y, rot, scale, color = state
        if scale != 1 or (x, y) not in checkpoint_set:
            continue
        if (x, y) in visited:
            continue
        if color != gc and gc not in (8, 9, 15):
            continue
        lower_bound = (
            (abs(x - gx) + 7) // 8
            + (abs(y - gy) + 3) // 4
            + int(rot != gr)
            + abs(scale - gs)
            + int(color != gc)
        )
        candidates.append((lower_bound, cost, path))
    if candidates:
        candidates.sort()
        return list(candidates[0][2])
    return None


def act(obs, available, memory):
    g = np.asarray(obs)
    panel = _find_panel(g)
    button_y, button_x = _find_button(g)

    if memory.pop("ran_program", False):
        memory.pop("plan", None)

    if panel is None:
        # Avoid issuing an invalid action if the panel is temporarily obscured.
        return "ACTION6", button_y, button_x

    panel_y, xs, bits, current = panel
    plan = memory.get("plan")

    if plan is not None and len(plan) != len(xs):
        memory.pop("plan", None)
        plan = None

    if plan is None:
        colors = (11, 9, 8, 15)
        goals = _find_pattern(g, _GOAL_TEMPLATES, colors)
        players = _find_pattern(g, _PLAYER_TEMPLATES, colors)

        if goals and players:
            # The actual goal is normally the smallest matching goal sprite.
            goals.sort(key=lambda p: (p[3], p[0], p[1]))
            outer = goals[0]
            gx, gy, gr, gs, gc = outer
            goal = (gx + gs, gy + gs, gr, gs, gc)

            # Prefer pieces on the same four-pixel lattice as the goal.
            players.sort(key=lambda p: (
                ((p[0] - goal[0]) % 4 != 0)
                + ((p[1] - goal[1]) % 4 != 0),
                -p[3],
                abs(p[0] - goal[0]) + abs(p[1] - goal[1]),
            ))
            player = players[0]

            visited = set(tuple(p) for p in memory.get("visited", []))
            visited.add((player[0], player[1]))
            memory["visited"] = [list(p) for p in visited]

            checkpoints = _find_checkpoints(g, player)
            walls = _wall_map(g, player)
            plan = _search(
                player, goal, current, bits, walls, checkpoints, visited
            )
            if plan is not None:
                memory["plan"] = plan

    if plan is None:
        # Running the current program is a safe fallback and can expose a
        # checkpoint or a changed state for the next observation.
        memory["ran_program"] = True
        return "ACTION6", button_y, button_x

    for i, (actual, desired) in enumerate(zip(current, plan)):
        difference = actual ^ desired
        if difference:
            bit = (difference & -difference).bit_length() - 1
            row = panel_y + 6 * (bit // 2) + (1 if bit % 2 == 0 else 4)
            col = xs[i] + 2
            return "ACTION6", int(row), int(col)

    memory["ran_program"] = True
    return "ACTION6", button_y, button_x

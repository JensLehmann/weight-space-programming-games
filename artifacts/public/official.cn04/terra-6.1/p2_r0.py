import time
from collections import deque


def _grid(obs):
    return [[int(obs[3 + 3 * y, 3 + 3 * x]) for x in range(20)]
            for y in range(20)]


def _components(grid, predicate):
    seen = set()
    result = []
    for y in range(20):
        for x in range(20):
            if (x, y) in seen or not predicate(grid[y][x]):
                continue
            q = deque([(x, y)])
            seen.add((x, y))
            cells = []
            while q:
                a, b = q.popleft()
                cells.append((a, b))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    c, d = a + dx, b + dy
                    if (0 <= c < 20 and 0 <= d < 20
                            and (c, d) not in seen
                            and predicate(grid[d][c])):
                        seen.add((c, d))
                        q.append((c, d))
            result.append(cells)
    return result


def _shape(cells, grid):
    x0 = min(x for x, y in cells)
    y0 = min(y for x, y in cells)
    x1 = max(x for x, y in cells)
    y1 = max(y for x, y in cells)
    mask = tuple(sorted((x - x0, y - y0) for x, y in cells))
    points = sorted((x - x0, y - y0) for x, y in cells
                    if grid[y][x] in (8, 3))
    ends = tuple((x, y, k) for k, (x, y) in enumerate(points))
    return x0, y0, (x1 - x0 + 1, y1 - y0 + 1, mask, ends)


def _rotate(shape):
    w, h, mask, ends = shape
    mask = tuple(sorted((h - 1 - y, x) for x, y in mask))
    ends = tuple(sorted((h - 1 - y, x, k) for x, y, k in ends))
    return h, w, mask, ends


def _same_shape(a, b):
    return (a[0] == b[0] and a[1] == b[1]
            and a[2] == b[2]
            and {(x, y) for x, y, _ in a[3]}
            == {(x, y) for x, y, _ in b[3]})


def _grab(mem, i, grid):
    """Read the selected piece, separating it from grey neighbours."""
    p = mem["pieces"][i]
    old = {(p["x"] + x, p["y"] + y) for x, y in p["shape"][2]}

    if mem["grey"]:
        comps = _components(
            grid, lambda c: c != mem["bg"] and c != 4
        )
    else:
        comps = _components(grid, lambda c: c != mem["bg"])

    if not comps:
        return None

    def score(cells):
        overlap = len(old.intersection(cells))
        x0 = min(x for x, y in cells)
        y0 = min(y for x, y in cells)
        dist = abs(x0 - p["x"]) + abs(y0 - p["y"])
        highlighted = sum(grid[y][x] in (8, 3) for x, y in cells)
        return 10 * overlap - 2 * dist + min(highlighted, 4)

    cells = max(comps, key=score)
    x, y, shape = _shape(cells, grid)
    p["x"], p["y"], p["shape"] = x, y, shape
    return shape


def _detect_selected(mem, grid):
    """Identify the highlighted piece in a grey-masked frame."""
    if not mem["grey"]:
        return None
    highlighted = {
        (x, y) for y in range(20) for x in range(20)
        if grid[y][x] != mem["bg"] and grid[y][x] != 4
        and grid[y][x] != 3
    }
    if not highlighted:
        return None

    best = None
    best_score = -10**9
    for i, p in enumerate(mem["pieces"]):
        mask = {(p["x"] + x, p["y"] + y)
                for x, y in p["shape"][2]}
        hit = len(mask & highlighted)
        if not hit:
            continue
        score = 4 * hit - 2 * (len(mask) - hit) - len(highlighted - mask)
        if score > best_score:
            best_score = score
            best = i
    return best


def _click_position(mem, i, grid):
    p = mem["pieces"][i]
    banned = mem.get("banned", {}).get(i, set())
    candidates = []
    for x, y in p["shape"][2]:
        gx, gy = p["x"] + x, p["y"] + y
        if not (0 <= gx < 20 and 0 <= gy < 20):
            continue
        if grid[gy][gx] == mem["bg"]:
            continue
        if (gx, gy) in banned:
            continue
        covered = 0
        for j, other in enumerate(mem["pieces"]):
            if j == i:
                continue
            ox, oy = gx - other["x"], gy - other["y"]
            if (ox, oy) in other["shape"][2]:
                covered += 1
        candidates.append((covered, gy, gx))
    if not candidates:
        return None
    _, y, x = min(candidates)
    return 3 + 3 * y, 3 + 3 * x


def _initialise(obs, memory):
    grid = _grid(obs)
    bg = int(obs[63, 63])
    comps = _components(grid, lambda c: c != bg)
    pieces = []
    for cells in comps:
        x, y, shape = _shape(cells, grid)
        pieces.append({"x": x, "y": y, "shape": shape})

    port_pieces = [
        i for i, c in enumerate(comps)
        if any(grid[y][x] in (8, 3) for x, y in c)
    ]
    grey_pieces = [
        i for i, c in enumerate(comps)
        if all(grid[y][x] in (4, 3) for x, y in c)
    ]
    grey = bool(grey_pieces and bg != 4)

    selected = None
    if len(port_pieces) == 1 and grey:
        selected = port_pieces[0]
    else:
        black = []
        for i, c in enumerate(comps):
            n = sum(grid[y][x] == 0 for x, y in c)
            if n:
                black.append((n, i))
        if black:
            selected = max(black)[1]

    memory.clear()
    memory.update({
        "bg": bg,
        "grey": grey,
        "pieces": pieces,
        "selected": selected,
        "phase": "scan",
        "scan_i": 0,
        "scan_stage": "select",
        "pending": None,
        "tried": set(),
        "plan": None,
        "banned": {},
        "fallback": 0,
    })


def _process_pending(mem, grid):
    pending = mem.get("pending")
    if pending is None:
        return
    mem["pending"] = None
    kind, i = pending[:2]
    p = mem["pieces"][i]

    if kind in ("scan_click", "plan_click"):
        actual = _detect_selected(mem, grid) if mem["grey"] else i
        mem["selected"] = actual
        if actual != i:
            if actual is not None and len(pending) >= 4:
                row, col = pending[2:4]
                gx, gy = (col - 3) // 3, (row - 3) // 3
                mem["banned"].setdefault(i, set()).add((gx, gy))
            return
        if kind == "scan_click":
            _grab(mem, i, grid)
            mem["scan_stage"] = "inspect"

    elif kind == "probe_first":
        new = _grab(mem, i, grid)
        if new is None:
            new = p["first"]
        if _same_shape(_rotate(p["first"]), new):
            opts = [new]
            for _ in range(3):
                opts.append(_rotate(opts[-1]))
            p["opts"] = opts
            p["cur"] = 0
            p["group"] = False
            p["shape"] = opts[0]
            mem["scan_i"] += 1
            mem["scan_stage"] = "select"
        else:
            p["group"] = True
            p["opts"] = [p["first"], new]
            p["seq"] = [0, 1]
            p["cur"] = 1
            mem["scan_stage"] = "probe_group"

    elif kind == "probe_group":
        new = _grab(mem, i, grid)
        if new is None:
            new = p["shape"]
        idx = None
        for k, old in enumerate(p["opts"]):
            if _same_shape(old, new):
                idx = k
                break
        if idx is None:
            idx = len(p["opts"])
            p["opts"].append(new)
        p["seq"].append(idx)
        p["cur"] = idx

        seq = p["seq"]
        pair = tuple(seq[-2:])
        repeated = any(tuple(seq[j:j + 2]) == pair
                       for j in range(len(seq) - 2))
        if repeated or len(seq) >= 24:
            edges = {}
            for j in range(len(seq) - 2):
                edges[(seq[j], seq[j + 1])] = seq[j + 2]
            p["edges"] = edges
            p["prev"] = seq[-2]
            p["shape"] = p["opts"][p["cur"]]
            mem["scan_i"] += 1
            mem["scan_stage"] = "select"
        else:
            mem["scan_stage"] = "probe_group"


def _turn_count(p, target):
    if p["cur"] == target:
        return 0
    if not p["group"]:
        return (target - p["cur"]) % 4
    prev, cur = p.get("prev", p["cur"]), p["cur"]
    for n in range(1, 30):
        nxt = p.get("edges", {}).get((prev, cur))
        if nxt is None:
            return 20
        prev, cur = cur, nxt
        if cur == target:
            return n
    return 20


def _turn_model(p):
    if not p["group"]:
        p["cur"] = (p["cur"] + 1) % 4
    else:
        prev, cur = p.get("prev", p["cur"]), p["cur"]
        nxt = p.get("edges", {}).get((prev, cur))
        if nxt is None:
            nxt = (cur + 1) % len(p["opts"])
        p["prev"], p["cur"] = cur, nxt
    p["shape"] = p["opts"][p["cur"]]
    w, h = p["shape"][:2]
    p["x"] = max(0, min(p["x"], 20 - w))
    p["y"] = max(0, min(p["y"], 20 - h))


def _median(values):
    values = sorted(values)
    return values[len(values) // 2]


def _search(mem):
    pieces = mem["pieces"]
    n = len(pieces)
    if not n or any("opts" not in p for p in pieces):
        return None

    root = max(
        range(n),
        key=lambda i: max(len(s[3]) for s in pieces[i]["opts"])
    )
    root_options = list(range(len(pieces[root]["opts"])))
    if pieces[root]["group"]:
        root_options.sort(
            key=lambda k: -len(pieces[root]["opts"][k][3])
        )

    deadline = time.monotonic() + 2.0
    nodes = [0]
    best = [None, float("inf")]
    placed = {}
    occupied = {}

    def bounds_ok():
        minx = min(v[0] for v in placed.values())
        miny = min(v[1] for v in placed.values())
        maxx = max(v[0] + pieces[i]["opts"][v[2]][0]
                   for i, v in placed.items())
        maxy = max(v[1] + pieces[i]["opts"][v[2]][1]
                   for i, v in placed.items())
        return maxx - minx <= 20 and maxy - miny <= 20

    def add(i, x, y, opt):
        shape = pieces[i]["opts"][opt]
        positions = [(x + ex, y + ey, eid) for ex, ey, eid in shape[3]]
        for px, py, eid in positions:
            if len(occupied.get((px, py), ())) >= 2:
                return False
        placed[i] = (x, y, opt)
        if not bounds_ok():
            del placed[i]
            return False
        for px, py, eid in positions:
            occupied.setdefault((px, py), []).append((i, eid))
        return True

    def remove(i):
        x, y, opt = placed.pop(i)
        for ex, ey, eid in pieces[i]["opts"][opt][3]:
            pos = (x + ex, y + ey)
            occupied[pos].pop()
            if not occupied[pos]:
                del occupied[pos]

    def evaluate():
        pairs = []
        for entries in occupied.values():
            if len(entries) != 2:
                return
            pairs.append(tuple(sorted(entries)))
        groups = tuple((i, placed[i][2]) for i in range(n)
                       if pieces[i]["group"])
        signature = (tuple(sorted(pairs)), groups)
        if signature in mem["tried"]:
            return

        minx = min(v[0] for v in placed.values())
        miny = min(v[1] for v in placed.values())
        maxx = max(v[0] + pieces[i]["opts"][v[2]][0]
                   for i, v in placed.items())
        maxy = max(v[1] + pieces[i]["opts"][v[2]][1]
                   for i, v in placed.items())
        sx = max(
            -minx,
            min(20 - maxx,
                _median([pieces[i]["x"] - placed[i][0]
                         for i in range(n)]))
        )
        sy = max(
            -miny,
            min(20 - maxy,
                _median([pieces[i]["y"] - placed[i][1]
                         for i in range(n)]))
        )

        cost = 0
        masks = []
        for i in range(n):
            x, y, opt = placed[i]
            p = pieces[i]
            shape = p["opts"][opt]
            cost += abs(x + sx - p["x"]) + abs(y + sy - p["y"])
            cost += _turn_count(p, opt)
            if p["group"]:
                largest = max(len(s[3]) for s in p["opts"])
                cost += 12 * (largest - len(shape[3]))
            masks.append({(x + a, y + b) for a, b in shape[2]})
        for i in range(n):
            for j in range(i):
                cost += 2 * len(masks[i] & masks[j])

        if cost < best[1]:
            target = [(placed[i][0] + sx, placed[i][1] + sy, placed[i][2])
                      for i in range(n)]
            best[:] = [(target, signature), cost]

    def dfs():
        nodes[0] += 1
        if nodes[0] % 128 == 0 and time.monotonic() >= deadline:
            return
        if nodes[0] > 180000:
            return

        open_positions = [pos for pos, entries in occupied.items()
                          if len(entries) == 1]
        remaining = [i for i in range(n) if i not in placed]
        if not remaining:
            if not open_positions:
                evaluate()
            return

        max_remaining = sum(max(len(s[3]) for s in pieces[i]["opts"])
                            for i in remaining)
        if len(open_positions) > max_remaining:
            return

        candidates = []
        if open_positions:
            anchor = min(open_positions)
            for i in remaining:
                p = pieces[i]
                for opt, shape in enumerate(p["opts"]):
                    for ex, ey, eid in shape[3]:
                        x, y = anchor[0] - ex, anchor[1] - ey
                        matches = 0
                        invalid = False
                        for ax, ay, _ in shape[3]:
                            entries = occupied.get((x + ax, y + ay), ())
                            if len(entries) >= 2:
                                invalid = True
                                break
                            if len(entries) == 1:
                                matches += 1
                        if invalid:
                            continue
                        if (len(open_positions) == max_remaining
                                and matches != len(shape[3])):
                            continue
                        distance = abs(x - (p["x"] - pieces[root]["x"]))
                        distance += abs(y - (p["y"] - pieces[root]["y"]))
                        candidates.append(
                            (-matches, distance, i, x, y, opt)
                        )
        else:
            i = remaining[0]
            for opt, shape in enumerate(pieces[i]["opts"]):
                for x, y in ((0, 0), (3, 0), (-3, 0), (0, 3),
                             (0, -3), (6, 0), (0, 6), (-6, 0), (0, -6)):
                    candidates.append(
                        (0, abs(x) + abs(y), i, x, y, opt)
                    )

        candidates.sort()
        seen = set()
        for _, _, i, x, y, opt in candidates:
            key = (i, x, y, opt)
            if key in seen:
                continue
            seen.add(key)
            if add(i, x, y, opt):
                dfs()
                remove(i)
            if time.monotonic() >= deadline or nodes[0] > 180000:
                return

    for opt in root_options:
        if time.monotonic() >= deadline:
            break
        if add(root, 0, 0, opt):
            dfs()
            remove(root)
    return best[0]


def act(obs, available, memory):
    if not memory or "pieces" not in memory:
        _initialise(obs, memory)

    mem = memory
    grid = _grid(obs)
    _process_pending(mem, grid)

    for _ in range(100):
        if mem["phase"] == "scan":
            i = mem["scan_i"]
            if i >= len(mem["pieces"]):
                mem["phase"] = "play"
                continue
            p = mem["pieces"][i]
            stage = mem["scan_stage"]

            if stage == "select":
                if mem["selected"] != i:
                    click = _click_position(mem, i, grid)
                    if click is None:
                        mem["banned"].pop(i, None)
                        click = _click_position(mem, i, grid)
                    if click is None:
                        mem["scan_i"] += 1
                        continue
                    mem["pending"] = ("scan_click", i, click[0], click[1])
                    return "ACTION6", click[0], click[1]
                _grab(mem, i, grid)
                mem["scan_stage"] = "inspect"
                continue

            if stage == "inspect":
                p["first"] = p["shape"]
                mem["pending"] = ("probe_first", i)
                return "ACTION5", None, None

            if stage == "probe_group":
                mem["pending"] = ("probe_group", i)
                return "ACTION5", None, None

        else:
            if mem["plan"] is None:
                result = _search(mem)
                if result is None:
                    mem["fallback"] += 1
                    directions = ("ACTION1", "ACTION4", "ACTION2", "ACTION3")
                    return directions[mem["fallback"] % 4], None, None
                target, signature = result
                mem["plan"] = target
                mem["signature"] = signature

            target = mem["plan"]
            unsolved = [
                i for i, p in enumerate(mem["pieces"])
                if (p["x"], p["y"], p["cur"]) != tuple(target[i])
            ]
            if not unsolved:
                mem["tried"].add(mem["signature"])
                mem["plan"] = None
                continue

            selected = mem["selected"]
            if selected in unsolved:
                i = selected
            else:
                choices = []
                for j in unsolved:
                    click = _click_position(mem, j, grid)
                    if click is not None:
                        choices.append((j, click))
                if not choices:
                    mem["banned"].clear()
                    for j in unsolved:
                        click = _click_position(mem, j, grid)
                        if click is not None:
                            choices.append((j, click))
                if not choices:
                    mem["tried"].add(mem["signature"])
                    mem["plan"] = None
                    continue
                i, click = choices[0]
                mem["pending"] = ("plan_click", i, click[0], click[1])
                return "ACTION6", click[0], click[1]

            p = mem["pieces"][i]
            tx, ty, opt = target[i]
            mem["banned"].clear()
            if p["cur"] != opt:
                _turn_model(p)
                return "ACTION5", None, None

            if p["x"] < tx:
                p["x"] += 1
                action = "ACTION4"
            elif p["x"] > tx:
                p["x"] -= 1
                action = "ACTION3"
            elif p["y"] < ty:
                p["y"] += 1
                action = "ACTION2"
            else:
                p["y"] -= 1
                action = "ACTION1"
            return action, None, None

    return "ACTION1", None, None

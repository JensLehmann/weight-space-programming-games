import time
import heapq
import itertools
from collections import deque, Counter
import numpy as np


# Piece identities: normal, horizontally mirrored, vertically mirrored, both.
_DIRMAP = (
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 2, 3),
    (1, 0, 3, 2),
)
_COLORS = {14: 0, 12: 1, 15: 2}
_ACTIONS = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")


def _geometry(obs):
    """Infer the cell lattice from the solid player sprites."""
    candidates = []
    for color in (10, 1):
        mask = obs == color
        seen = set()
        for y, x in zip(*np.where(mask)):
            y, x = int(y), int(x)
            if (y, x) in seen or y in (0, 63):
                continue
            stack = [(y, x)]
            seen.add((y, x))
            pts = []
            while stack:
                yy, xx = stack.pop()
                pts.append((yy, xx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = yy + dy, xx + dx
                    if (0 < ny < 63 and 0 <= nx < 64
                            and mask[ny, nx] and (ny, nx) not in seen):
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            h, w = max(ys) - min(ys) + 1, max(xs) - min(xs) + 1
            s = min(h, w)
            if 2 <= s <= 12 and h % s == 0 and w % s == 0:
                if len(pts) == h * w:
                    candidates.append((s, min(xs) % s, min(ys) % s))
        if candidates:
            break
    if candidates:
        s, ox, oy = Counter(candidates).most_common(1)[0][0]
    else:
        s, ox, oy = 4, 2, 2
    return s, ox, oy, (64 - ox) // s, (64 - oy) // s


def _wall_image(obs):
    out = np.empty((64, 64), dtype=np.int16)
    for y0, y1 in ((1, 32), (32, 63)):
        for x0, x1 in ((0, 32), (32, 64)):
            a = obs[y0:y1, x0:x1].ravel()
            a = a[a != 5]
            c = int(np.bincount(a, minlength=16).argmax()) if len(a) else 0
            out[0 if y0 == 1 else 32:32 if y1 == 32 else 64,
                x0:x1] = c
    return out


def _read_board(obs, memory):
    if "geom" not in memory:
        memory["geom"] = _geometry(obs)
    s, ox, oy, W, H = memory["geom"]
    N = W * H
    old = memory.get("board")
    walls = set(old["walls"]) if old else set()
    traps = set(old["traps"]) if old else set()
    gates = dict(old["gates"]) if old else {}
    switches = dict(old["switches"]) if old else {}
    extras = set(old["extras"]) if old else set()

    wallpic = _wall_image(obs)
    players, boxes, selected = [], [], None
    marks = {}
    visible_gates = set()

    for y in range(H):
        for x in range(W):
            p = y * W + x
            xx, yy = ox + x * s, oy + y * s
            a = obs[yy:yy+s, xx:xx+s]
            e = wallpic[yy:yy+s, xx:xx+s]
            # The action counter overwrites the first and last display rows.
            lo = 1 if yy == 0 else 0
            hi = s - 1 if yy + s == 64 else s
            v, ev = a[lo:hi], e[lo:hi]
            if not v.size:
                continue

            is_box = False
            if s >= 3:
                border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
                inner = a[1:-1, 1:-1]
                if np.all(border == 5) and inner.size:
                    if np.all(inner == 9) or np.all(inner == 11):
                        is_box = True
                        boxes.append(p)
                        if np.all(inner == 11):
                            selected = p

            if is_box:
                walls.discard(p)
                extras.discard(p)
                continue

            if np.mean(v == ev) > 0.97:
                walls.add(p)
                extras.discard(p)
                continue

            walls.discard(p)
            if np.all(v == 10) or np.all(v == 1):
                players.append(p)
                extras.discard(p)
            elif np.all(v == 5):
                extras.discard(p)
            elif np.all(v == v.flat[0]) and int(v.flat[0]) in _COLORS:
                marks[p] = _COLORS[int(v.flat[0])]
                extras.discard(p)
            else:
                nonfloor = v[v != 5]
                floor_count = int(np.sum(v == 5))
                if (nonfloor.size and floor_count
                        and np.all(nonfloor == nonfloor.flat[0])
                        and abs(floor_count - nonfloor.size) <= s + 1):
                    traps.add(p)
                    extras.discard(p)
                elif np.all(v == 9) or np.all(v == 11):
                    boxes.append(p)
                    if np.all(v == 11):
                        selected = p
                    extras.discard(p)
                else:
                    extras.add(p)

    # Multi-cell colored objects are gates; single cells are switches.
    seen = set()
    for p, color in marks.items():
        if p in seen:
            continue
        component = []
        todo = [p]
        seen.add(p)
        while todo:
            q = todo.pop()
            component.append(q)
            x, y = q % W, q // W
            for r in (q-W if y else -1,
                      q+W if y+1 < H else -1,
                      q-1 if x else -1,
                      q+1 if x+1 < W else -1):
                if r in marks and marks[r] == color and r not in seen:
                    seen.add(r)
                    todo.append(r)
        is_gate = len(component) >= 2 or any(q in gates for q in component)
        for q in component:
            if is_gate:
                gates[q] = color
                switches.pop(q, None)
                visible_gates.add(q)
            else:
                switches[q] = color

    # Preserve blocker identities when one has moved.
    previous = memory.get("box_positions", [])
    remaining = set(boxes)
    ordered = [None] * len(previous)
    if len(previous) == len(boxes):
        for i, p in enumerate(previous):
            if p in remaining:
                ordered[i] = p
                remaining.remove(p)
        for i, p in enumerate(previous):
            if ordered[i] is None and remaining:
                q = min(remaining, key=lambda q:
                        abs(q % W - p % W) + abs(q // W - p // W))
                ordered[i] = q
                remaining.remove(q)
        boxes = ordered
    else:
        boxes = sorted(boxes)
    boxes = tuple(boxes)
    mode = boxes.index(selected) if selected in boxes else -1

    open_now = 0
    for c in set(gates.values()):
        if not any(gates[p] == c for p in visible_gates):
            open_now |= 1 << c

    signature = (tuple(sorted(walls)), tuple(sorted(traps)),
                 tuple(sorted(gates.items())), tuple(sorted(switches.items())),
                 tuple(sorted(extras)))
    changed = old is None or old["signature"] != signature

    board = dict(W=W, H=H, N=N, s=s, ox=ox, oy=oy,
                 walls=walls, traps=traps, gates=gates, switches=switches,
                 extras=extras, signature=signature, open_now=open_now)
    if changed:
        solid = walls | extras
        nbr = [[0] * N for _ in range(4)]
        for p in range(N):
            x, y = p % W, p // W
            qs = (p-W if y else p, p+W if y+1 < H else p,
                  p-1 if x else p, p+1 if x+1 < W else p)
            for d, q in enumerate(qs):
                nbr[d][p] = p if q in solid else q
        board["nbr"] = nbr
        memory.pop("plan", None)
        memory["tables"] = {}
    else:
        board["nbr"] = old["nbr"]
    memory["board"] = board
    memory["box_positions"] = boxes
    return board, tuple(sorted(set(players))), boxes, mode


def _open_mask(pieces, board):
    mask = 0
    switches = board["switches"]
    for p in pieces:
        if p in switches:
            mask |= 1 << switches[p]
    return mask


def _advance(pieces, direction, boxes, board, frozen=(), open_override=None):
    """Exact player movement and pair merging, excluding hazard resets."""
    W = board["W"]
    old = pieces
    new = list(old)
    active = [i for i, p in enumerate(old) if p >= 0]
    opened = _open_mask(old, board) if open_override is None else open_override
    gates, traps, nbr = board["gates"], board["traps"], board["nbr"]
    blocked = set(boxes)
    for i in active:
        p = old[i]
        q = nbr[_DIRMAP[i][direction]][p]
        if q in blocked or (q in gates and not (opened & (1 << gates[q]))):
            q = p
        if q in traps:
            return None
        new[i] = q

    # The simulator also merges horizontally adjacent pieces that cross.
    for ii, i in enumerate(active):
        for j in active[ii+1:]:
            a, b = old[i], old[j]
            if abs(a-b) == 1 and a // W == b // W:
                if new[i] == b or new[j] == a:
                    x = (new[i] % W + new[j] % W) // 2
                    y = (new[i] // W + new[j] // W) // 2
                    new[i] = new[j] = y * W + x

    groups = {}
    for i in active:
        groups.setdefault(new[i], []).append(i)
    done = set(frozen)
    for p, inds in groups.items():
        if len(inds) >= 2:
            done.add(p)
            new[inds[0]] = new[inds[1]] = -1
            for i in inds[2:]:
                new[i] = old[i]
    return tuple(new), tuple(sorted(done))


def _visible(hypothesis):
    p, frozen = hypothesis
    return tuple(sorted(set(q for q in p if q >= 0) | set(frozen)))


def _initial_hypotheses(points, W):
    points = tuple(points)
    result = []
    if len(points) >= 4:
        preferred = sorted(points[:4], key=lambda p: (-p // W, p % W))
        for perm in itertools.permutations(preferred):
            result.append((tuple(perm), ()))
    elif len(points) == 3:
        for frozen in points:
            active = [p for p in points if p != frozen]
            for roles in itertools.combinations(range(4), 2):
                for vals in (active, active[::-1]):
                    p = [-1] * 4
                    p[roles[0]], p[roles[1]] = vals
                    result.append((tuple(p), (frozen,)))
    elif len(points) == 2:
        a, b = sorted(points, key=lambda p: p % W)
        result = [((a, b, -1, -1), ()), ((b, a, -1, -1), ())]
    elif len(points) == 1:
        result = [((points[0], points[0], -1, -1), ())]
    return result


def _pair_tables(board, boxes, relaxed_boxes, deadline, memory):
    """Reverse pair-state searches provide obstacle-aware search guidance."""
    key = (tuple(sorted(boxes)) if not relaxed_boxes else (), relaxed_boxes)
    cache = memory.setdefault("tables", {})
    if key in cache:
        return cache[key]

    N, W, H = board["N"], board["W"], board["H"]
    blocked = board["walls"] | board["extras"] | board["traps"]
    if not relaxed_boxes:
        blocked = blocked | set(boxes)
    free = [p for p in range(N) if p not in blocked]
    free_set = set(free)
    pred = [[[] for _ in range(N)] for _ in range(4)]
    gates = board["gates"]

    for p in free:
        x, y = p % W, p // W
        qs = (p-W if y else p, p+W if y+1 < H else p,
              p-1 if x else p, p+1 if x+1 < W else p)
        for d, q in enumerate(qs):
            if q not in free_set:
                q = p
            pred[d][q].append(p)
            # Optional stopping relaxes the state of gates and movable blockers.
            if q != p and (relaxed_boxes or q in gates):
                pred[d][p].append(p)

    tables = {}
    for relative in (1, 2, 3):
        if time.monotonic() > deadline:
            return None
        dist = [-1] * (N * N)
        queue = deque()
        for p in free:
            z = p * N + p
            dist[z] = 0
            queue.append(z)
        for p in free:
            if p % W + 1 < W and p + 1 in free_set:
                for a, b in ((p, p+1), (p+1, p)):
                    z = a * N + b
                    if dist[z] < 0:
                        dist[z] = 1
                        queue.append(z)

        mapping = _DIRMAP[relative]
        count = 0
        while queue:
            z = queue.popleft()
            a, b = divmod(z, N)
            nd = dist[z] + 1
            for d in range(4):
                pa = pred[d][a]
                pb = pred[mapping[d]][b]
                for aa in pa:
                    base = aa * N
                    for bb in pb:
                        zz = base + bb
                        if dist[zz] < 0:
                            dist[zz] = nd
                            queue.append(zz)
            count += 1
            if count % 2048 == 0 and time.monotonic() > deadline:
                return None
        tables[relative] = dist

    if len(cache) >= 3:
        cache.clear()
    cache[key] = tables
    return tables


def _heuristic(p, tables, N, W):
    active = [i for i in range(4) if p[i] >= 0]
    if not active:
        return 0.0
    if tables is None:
        def distance(i, j):
            return (abs(p[i] % W - p[j] % W)
                    + abs(p[i] // W - p[j] // W)) / 2
    else:
        def distance(i, j):
            v = tables[i ^ j][p[i] * N + p[j]]
            return 10000 if v < 0 else v

    if len(active) == 2:
        return float(distance(*active))
    if len(active) == 4:
        choices = []
        for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
            x, y = distance(a, b), distance(c, d)
            choices.append(max(x, y) + 0.22 * min(x, y))
        return float(min(choices))
    return 0.0


def _search(pieces, boxes, mode, board, tables, joint, deadline):
    N, W = board["N"], board["W"]
    hcache = {}

    def heuristic(p):
        if p not in hcache:
            hcache[p] = _heuristic(p, tables, N, W)
        return hcache[p]

    root = (pieces, boxes, mode if joint else -1)
    h0 = heuristic(pieces)
    if h0 >= 9999:
        return None, True
    weight = 1.9 if joint else (1.6 if sum(p >= 0 for p in pieces) == 4 else 1.2)
    heap = [(weight * h0, h0, 0, root)]
    costs = {root: 0}
    parents = {}
    best = root
    best_key = (h0, 0)
    expanded = 0

    def reconstruct(state):
        path = []
        while state != root:
            prev, who, d = parents[state]
            path.append((who, d))
            state = prev
        path.reverse()
        return path

    while heap:
        _, h, g, state = heapq.heappop(heap)
        if costs.get(state) != g:
            continue
        p, b, selected = state
        if all(q < 0 for q in p):
            return reconstruct(state), True

        expanded += 1
        if expanded % 128 == 0:
            if time.monotonic() >= deadline or expanded >= 180000:
                return reconstruct(best) or None, False

        successors = []
        for d in range(4):
            override = board["open_now"] if state == root else None
            moved = _advance(p, d, b, board, open_override=override)
            if moved is None:
                continue
            npieces = moved[0]
            if npieces != p:
                successors.append(((npieces, b, -1),
                                   1 + (selected != -1), -1, d))

        if joint and b:
            occupied_players = set(q for q in p if q >= 0)
            opened = board["open_now"] if state == root else _open_mask(p, board)
            for i, q in enumerate(b):
                # Blockers can be prepared when approached, then moved as far
                # as necessary while selected.
                near = selected == i or any(
                    abs(q % W - z % W) + abs(q // W - z // W) <= 2
                    for z in occupied_players)
                if not near:
                    continue
                for d in range(4):
                    r = board["nbr"][d][q]
                    if r == q or r in b or r in occupied_players:
                        continue
                    if r in board["traps"] or r in board["switches"]:
                        continue
                    c = board["gates"].get(r)
                    if c is not None and not (opened & (1 << c)):
                        continue
                    nb = list(b)
                    nb[i] = r
                    successors.append(((p, tuple(nb), i),
                                       1 + (selected != i), i, d))

        for nxt, cost, who, d in successors:
            ng = g + cost
            if ng > 145 or ng >= costs.get(nxt, 10**9):
                continue
            nh = heuristic(nxt[0])
            if nh >= 9999:
                continue
            costs[nxt] = ng
            parents[nxt] = (state, who, d)
            heapq.heappush(heap, (ng + weight * nh, nh, ng, nxt))
            k = (nh + 0.015 * ng, ng)
            if k < best_key:
                best_key, best = k, nxt

    return None, True


def _make_plan(hypothesis, boxes, mode, board, memory, deadline):
    p = hypothesis[0]
    now = time.monotonic()
    tables = _pair_tables(board, boxes, False,
                          min(deadline, now + 2.0), memory)
    fixed_deadline = min(deadline, time.monotonic() + (2.0 if boxes else 7.5))
    path, complete = _search(p, boxes, -1, board, tables, False, fixed_deadline)
    if path and complete:
        return path
    if not boxes:
        return path

    partial = path
    if time.monotonic() < deadline - 0.3:
        tables = _pair_tables(board, (), True,
                              min(deadline - 0.15, time.monotonic() + 2.0),
                              memory)
        path, _ = _search(p, boxes, mode, board, tables, True, deadline)
        if path:
            return path
    return partial


def _click_coordinates(p, board):
    row = board["oy"] + (p // board["W"]) * board["s"] + board["s"] // 2
    col = board["ox"] + (p % board["W"]) * board["s"] + board["s"] // 2
    return int(min(63, row)), int(min(63, col))


def _emit(memory, board, hypotheses, favorite, boxes,
          action, direction=None, who=-1, click=None):
    expected = []
    next_favorite = None
    if direction is not None and who == -1:
        for h in hypotheses:
            nxt = _advance(h[0], direction, boxes, board, h[1],
                           board["open_now"])
            if nxt is not None:
                expected.append(nxt)
                if h == favorite:
                    next_favorite = nxt
    else:
        expected = list(hypotheses)
        next_favorite = favorite

    memory["expected"] = list(dict.fromkeys(expected))
    memory["favorite"] = next_favorite
    next_boxes = list(boxes)
    if direction is not None and who >= 0 and who < len(next_boxes):
        q = next_boxes[who]
        next_boxes[who] = board["nbr"][direction][q]
    memory["box_positions"] = tuple(next_boxes)

    if action == "ACTION6":
        row, col = _click_coordinates(click, board)
        return action, row, col
    return action, None, None


def act(obs, available, memory):
    start = time.monotonic()
    deadline = start + 8.8
    obs = np.asarray(obs)
    board, points, boxes, mode = _read_board(obs, memory)

    expected = memory.get("expected", [])
    hypotheses = [h for h in expected if _visible(h) == points]
    previous_favorite = memory.get("favorite")
    if not hypotheses:
        hypotheses = _initial_hypotheses(points, board["W"])
        memory.pop("plan", None)
        memory["probes"] = 0
    elif previous_favorite not in hypotheses:
        memory.pop("plan", None)

    if not hypotheses:
        action = "ACTION5" if "ACTION5" in available else available[0]
        return action, None, None

    favorite = previous_favorite if previous_favorite in hypotheses else hypotheses[0]
    memory["favorite"] = favorite

    def clear_cell():
        if points:
            return points[0]
        for q in range(board["N"]):
            if q not in boxes and q not in board["walls"]:
                return q
        return 0

    def emit(action, direction=None, who=-1, click=None):
        return _emit(memory, board, hypotheses, favorite, boxes,
                     action, direction, who, click)

    # Identify the mirror assignments with safe, informative observations.
    if (not memory.get("plan") and len(hypotheses) > 1
            and memory.get("probes", 0) < 8):
        informative = []
        for d in range(4):
            if _ACTIONS[d] not in available:
                continue
            groups = Counter()
            safe = True
            for h in hypotheses:
                nxt = _advance(h[0], d, boxes, board, h[1], board["open_now"])
                if nxt is None:
                    safe = False
                    break
                groups[_visible(nxt)] += 1
            if safe and len(groups) > 1:
                ambiguity = sum(v*v for v in groups.values())
                informative.append((ambiguity, 0 if d == 3 else 1, d))
        if informative:
            if mode != -1 and "ACTION6" in available:
                return emit("ACTION6", click=clear_cell())
            d = min(informative)[2]
            memory["probes"] = memory.get("probes", 0) + 1
            return emit(_ACTIONS[d], d)

    plan = memory.get("plan")
    if not plan:
        plan = _make_plan(favorite, boxes, mode, board, memory, deadline)
        memory["plan"] = plan or []

    if plan:
        who, d = plan[0]
        if who != mode:
            if "ACTION6" in available:
                cell = clear_cell() if who == -1 else boxes[who]
                return emit("ACTION6", click=cell)
            memory["plan"] = []
        elif _ACTIONS[d] in available:
            memory["plan"] = plan[1:]
            return emit(_ACTIONS[d], d, who)

    # Bounded-search fallback: favor previously unvisited safe configurations.
    if mode != -1 and "ACTION6" in available:
        return emit("ACTION6", click=clear_cell())

    visits = memory.setdefault("visits", {})
    current_key = favorite[0], boxes
    visits[current_key] = visits.get(current_key, 0) + 1
    choices = []
    for d in range(4):
        if _ACTIONS[d] not in available:
            continue
        nxt = _advance(favorite[0], d, boxes, board, favorite[1],
                       board["open_now"])
        if nxt is not None and nxt[0] != favorite[0]:
            key = nxt[0], boxes
            h = _heuristic(nxt[0], None, board["N"], board["W"])
            choices.append((visits.get(key, 0), h, d))
    if choices:
        d = min(choices)[2]
        return emit(_ACTIONS[d], d)

    # If completely blocked, select a blocker that has a legal escape.
    if boxes and "ACTION6" in available:
        occupied = set(q for q in favorite[0] if q >= 0)
        for i, p in enumerate(boxes):
            for d in range(4):
                q = board["nbr"][d][p]
                if (q != p and q not in boxes and q not in occupied
                        and q not in board["traps"] and q not in board["switches"]
                        and (q not in board["gates"]
                             or board["open_now"] & (1 << board["gates"][q]))):
                    memory["plan"] = [(i, d)]
                    return emit("ACTION6", click=p)

    action = "ACTION5" if "ACTION5" in available else next(
        (a for a in _ACTIONS if a in available), available[0])
    return emit(action)

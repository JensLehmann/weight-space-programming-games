import numpy as np
import collections
import itertools
import heapq
import math
import time

DIR = ((0, -1), (1, 0), (0, 1), (-1, 0))
SPECIAL = {2, 3, 4, 5, 13, 15}
INF = 120


def rotate(x, y, k):
    k %= 4
    if k == 0:
        return x, y
    if k == 1:
        return -y, x
    if k == 2:
        return -x, -y
    return y, -x


def components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        todo, pts = [(y, x)], []
        while todo:
            yy, xx = todo.pop()
            pts.append((yy, xx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = yy + dy, xx + dx
                if (0 <= ny < h and 0 <= nx < w and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    todo.append((ny, nx))
        result.append(pts)
    return result


def read_buttons(obs):
    buttons = []
    ignore = np.zeros((64, 64), dtype=bool)
    for pts in components(obs == 2):
        y = min(p[0] for p in pts)
        x = min(p[1] for p in pts)
        h = max(p[0] for p in pts) - y + 1
        w = max(p[1] for p in pts) - x + 1
        if min(h, w) < 5:
            continue
        counts = np.bincount(obs[y:y+h, x:x+w].ravel(), minlength=16)
        for c in SPECIAL:
            counts[c] = 0
        c = int(np.argmax(counts))
        if not counts[c]:
            continue
        ignore[y:y+h, x:x+w] = True
        if w == h:
            buttons.append((c, 0, y+h//2, x+w//2))
        elif w > h:
            buttons.append((c, -1, y+h//2, x+1))
            buttons.append((c, 1, y+h//2, x+w-2))
        else:
            buttons.append((c, -1, y+1, x+w//2))
            buttons.append((c, 1, y+h-2, x+w//2))
    ignore[63] = True
    return buttons, ignore


def read_markers(obs, ignore):
    goals = []
    cross_pixels = set()
    for y in range(1, 63):
        for x in range(1, 63):
            if ignore[y, x]:
                continue
            arms = ((y-1, x), (y+1, x), (y, x-1), (y, x+1))
            corners = ((y-1, x-1), (y-1, x+1),
                       (y+1, x-1), (y+1, x+1))
            if (all(obs[a, b] == 13 for a, b in arms) and
                    all(obs[a, b] != 13 for a, b in corners)):
                goals.append((x-1, y-1))
                cross_pixels.update(arms)
    dots = []
    for y, x in zip(*np.nonzero((obs == 13) & ~ignore)):
        y, x = int(y), int(x)
        if (y, x) not in cross_pixels:
            dots.append((x-1, y-1))
    return sorted(set(dots)), sorted(set(goals))


def find_lattice(obs, ignore):
    a = obs.copy()
    a[ignore] = 5
    colors = [c for c in range(16) if c not in SPECIAL]
    best = None
    for ry in range(3):
        for rx in range(3):
            score = 0
            for y in range(ry-3, 64, 3):
                for x in range(rx-3, 64, 3):
                    block = a[max(0, y):min(64, y+3),
                              max(0, x):min(64, x+3)]
                    if block.size:
                        cnt = np.bincount(block.ravel(), minlength=16)
                        score += sum(int(cnt[c])**2 for c in colors)
            key = (score, -rx-ry, -ry, -rx)
            if best is None or key > best[0]:
                best = (key, rx, ry)
    return best[1], best[2], a


def split_bars(cells):
    left, bars = set(cells), []
    while left:
        best = None
        for x, y in left:
            if (x-3, y) not in left:
                run, xx = [], x
                while (xx, y) in left:
                    run.append((xx, y))
                    xx += 3
                key = (len(run), 1, -y, -x)
                if best is None or key > best[0]:
                    best = (key, run)
            if (x, y-3) not in left:
                run, yy = [], y
                while (x, yy) in left:
                    run.append((x, yy))
                    yy += 3
                key = (len(run), 0, -y, -x)
                if best is None or key > best[0]:
                    best = (key, run)
        run = best[1]
        left.difference_update(run)
        bars.append(run)
    return bars


def read_shapes(obs, ignore):
    rx, ry, a = find_lattice(obs, ignore)
    labels = {}
    for y in range(ry, 62, 3):
        for x in range(rx, 62, 3):
            cnt = np.bincount(a[y:y+3, x:x+3].ravel(), minlength=16)
            for c in SPECIAL:
                cnt[c] = 0
            color = int(np.argmax(cnt))
            if cnt[color]:
                labels[x, y] = color

    used, parts = set(), []
    for cell, color in labels.items():
        if cell in used:
            continue
        used.add(cell)
        todo, cells = [cell], []
        while todo:
            x, y = todo.pop()
            cells.append((x, y))
            for dx, dy in ((3, 0), (-3, 0), (0, 3), (0, -3)):
                p = x+dx, y+dy
                if p not in used and labels.get(p) == color:
                    used.add(p)
                    todo.append(p)
        if color == 1:
            parts.extend((color, bar) for bar in split_bars(cells))
        else:
            parts.append((color, cells))

    shapes = []
    for color, cells in parts:
        x = min(p[0] for p in cells)
        y = min(p[1] for p in cells)
        w = max(p[0] for p in cells)-x+3
        h = max(p[1] for p in cells)-y+3
        if min(w, h) != 3:
            shapes.append(dict(c=color, x=x, y=y, w=w, h=h, odd=True))
            continue
        if color == 1:
            o, cap = (1 if w >= h else 2), False
        else:
            scores = [-1]*4
            if w == 3:
                scores[0] = int(np.count_nonzero(a[y+h-1, x:x+3] == 3))
                scores[2] = int(np.count_nonzero(a[y, x:x+3] == 3))
            if h == 3:
                scores[1] = int(np.count_nonzero(a[y:y+3, x] == 3))
                scores[3] = int(np.count_nonzero(a[y:y+3, x+w-1] == 3))
            o = max(range(4), key=lambda k: scores[k])
            cap = scores[o] > 0
            if not cap:
                o = 1 if w >= h else 2
        ax, ay = x, y
        if o == 0:
            ay += h-3
        elif o == 3:
            ax += w-3
        shapes.append(dict(c=color, x=x, y=y, w=w, h=h,
                           ax=ax, ay=ay, n=max(w, h)//3,
                           o=o, cap=cap, odd=False))
    return shapes, rx, ry


def box(ax, ay, n, o):
    dx, dy = DIR[o]
    ex, ey = ax+3*(n-1)*dx, ay+3*(n-1)*dy
    return min(ax, ex), min(ay, ey), abs(ex-ax)+3, abs(ey-ay)+3


def overlap(a, b):
    x, y, w, h = a
    xx, yy, ww, hh = b
    return x < xx+ww and xx < x+w and y < yy+hh and yy < y+h


def prefix(mask):
    p = np.zeros((65, 65), dtype=np.int32)
    p[1:, 1:] = mask.astype(np.int32).cumsum(0).cumsum(1)
    return p.tolist()


def occupied(p, b, bounded=True):
    x, y, w, h = b
    if bounded and (x < 0 or y < 0 or x+w > 64 or y+h > 64):
        return True
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(64, x+w), min(64, y+h)
    if x1 >= x2 or y1 >= y2:
        return False
    return p[y2][x2]-p[y1][x2]-p[y2][x1]+p[y1][x1] != 0


def geometry(m, state):
    geo = [None]*len(m["shapes"])
    shapes = m["shapes"]
    for i in m["order"]:
        s = shapes[i]
        p, q = m["parent"][i], m["param"][i]
        n = state[2*q] if q >= 0 else s["n"]
        rel = state[2*q+1] if q >= 0 else m["relative"][i]
        if p == -1:
            ax, ay = m["offsets"][i]
            o = rel
        else:
            px, py, pn, po, _ = geo[p]
            vx, vy = m["offsets"][i]
            vx, vy = rotate(vx, vy, po-shapes[p]["o"])
            dx, dy = DIR[po]
            ax, ay = px+3*(pn-1)*dx+vx, py+3*(pn-1)*dy+vy
            o = (po+rel) % 4
        geo[i] = (ax, ay, n, o, box(ax, ay, n, o))

    marks = []
    for p, vx, vy in m["markers"]:
        if p == -1:
            marks.append((vx, vy))
        else:
            ax, ay, n, o, _ = geo[p]
            vx, vy = rotate(vx, vy, o-shapes[p]["o"])
            dx, dy = DIR[o]
            marks.append((ax+3*(n-1)*dx+vx, ay+3*(n-1)*dy+vy))
    return geo, marks


def valid(m, geo):
    boxes = [g[4] for g in geo]
    outside = m["outside"]
    for i, b in enumerate(boxes):
        if occupied(m["prefix"], b, i not in outside):
            return False
        for j in range(i):
            if overlap(b, boxes[j]):
                return False
    return True


def transition(m, state, action):
    color, kind, _, _ = m["buttons"][action]
    out, changed = list(state), False
    for i, s in enumerate(m["shapes"]):
        q = m["param"][i]
        if q < 0 or s["c"] != color:
            continue
        if kind:
            old = out[2*q]
            new = max(1, old+kind)
            if new > 21:
                return None
            out[2*q] = new
            changed |= new != old
        else:
            r = out[2*q+1]
            out[2*q+1] = (1 if m["parent"][i] != -1 and r == 3
                          else (r-1) % 4)
            changed = True
    return tuple(out) if changed else None


def goal_distances(m):
    rx, ry = m["rx"], m["ry"]
    usable = {(x, y) for y in range(ry, 62, 3)
              for x in range(rx, 62, 3)
              if not occupied(m["prefix"], (x, y, 3, 3))}
    result = []
    for target in m["goals"]:
        d = {target: 0}
        todo = collections.deque([target])
        while todo:
            x, y = todo.popleft()
            for dx, dy in ((3, 0), (-3, 0), (0, 3), (0, -3)):
                p = x+dx, y+dy
                if p in usable and p not in d:
                    d[p] = d[x, y]+1
                    todo.append(p)
        result.append(d)
    return result


def allowed_assignments(m, state):
    _, marks = geometry(m, state)
    allowed = [[True]*len(m["goals"]) for _ in marks]
    for i, (p, vx, vy) in enumerate(m["markers"]):
        if p == -1:
            allowed[i] = [g == marks[i] for g in m["goals"]]
            continue
        if m["parent"][p] != -1:
            continue
        s = m["shapes"][p]
        if s["c"] in m["rot_colors"]:
            continue
        dx, dy = DIR[s["o"]]
        for j, goal in enumerate(m["goals"]):
            if s["c"] not in m["length_colors"]:
                allowed[i][j] = goal == marks[i]
                continue
            xx, yy = goal[0]-marks[i][0], goal[1]-marks[i][1]
            distance = xx*dx+yy*dy
            new_n = s["n"]+distance//3
            allowed[i][j] = (xx*dy == yy*dx and distance % 3 == 0
                              and 1 <= new_n <= 21)

    if len(marks) == len(m["goals"]):
        changed = True
        while changed:
            changed = False
            for i, row in enumerate(allowed):
                choices = [j for j, ok in enumerate(row) if ok]
                if len(choices) == 1:
                    j = choices[0]
                    for k in range(len(marks)):
                        if k != i and allowed[k][j]:
                            allowed[k][j] = False
                            changed = True
    return allowed


def make_model(obs):
    buttons, ignore = read_buttons(obs)
    dots, goals = read_markers(obs, ignore)
    shapes, rx, ry = read_shapes(obs, ignore)
    fixed = (obs == 15) & ~ignore
    clean = []
    for s in shapes:
        if s["odd"]:
            region = obs[s["y"]:s["y"]+s["h"],
                         s["x"]:s["x"]+s["w"]]
            fixed[s["y"]:s["y"]+s["h"],
                  s["x"]:s["x"]+s["w"]] |= region == s["c"]
        else:
            clean.append(s)
    shapes = clean
    n = len(shapes)
    parent = [-1]*n

    contacts = []
    for i, s in enumerate(shapes):
        if not s["cap"]:
            continue
        dx, dy = DIR[s["o"]]
        nx, ny = s["ax"]+3*s["n"]*dx, s["ay"]+3*s["n"]*dy
        for j, t in enumerate(shapes):
            if i == j:
                continue
            if t["cap"]:
                if (nx, ny) == (t["ax"], t["ay"]):
                    parallel = (s["o"]-t["o"]) % 2 == 0
                    contacts.append((int(parallel), i, j, None))
            else:
                endpoints = [(t["x"], t["y"],
                              1 if t["w"] >= t["h"] else 2)]
                if t["w"] > 3:
                    endpoints.append((t["x"]+t["w"]-3, t["y"], 3))
                elif t["h"] > 3:
                    endpoints.append((t["x"], t["y"]+t["h"]-3, 0))
                for ex, ey, eo in endpoints:
                    if (nx, ny) == (ex, ey):
                        parallel = (s["o"]-eo) % 2 == 0
                        contacts.append((int(parallel), i, j, eo))

    for _, i, j, support_o in sorted(contacts):
        if parent[j] != -1:
            continue
        p, cyclic = i, False
        while p != -1:
            if p == j:
                cyclic = True
                break
            p = parent[p]
        if cyclic:
            continue
        parent[j] = i
        if support_o is not None:
            t = shapes[j]
            t["o"] = support_o
            t["ax"], t["ay"] = t["x"], t["y"]
            if support_o == 0:
                t["ay"] += t["h"]-3
            elif support_o == 3:
                t["ax"] += t["w"]-3

    order, pending = [], set(range(n))
    while pending:
        ready = sorted(i for i in pending
                       if parent[i] == -1 or parent[i] in order)
        if not ready:
            i = min(pending)
            parent[i] = -1
            ready = [i]
        for i in ready:
            order.append(i)
            pending.remove(i)

    rot_colors = {b[0] for b in buttons if b[1] == 0}
    length_colors = {b[0] for b in buttons if b[1] != 0}
    controlled = rot_colors | length_colors
    param, state = [-1]*n, []
    offsets, relative = [], []
    for i, s in enumerate(shapes):
        p = parent[i]
        if p == -1:
            offsets.append((s["ax"], s["ay"]))
            relative.append(s["o"])
        else:
            t = shapes[p]
            dx, dy = DIR[t["o"]]
            offsets.append((s["ax"]-t["ax"]-3*(t["n"]-1)*dx,
                            s["ay"]-t["ay"]-3*(t["n"]-1)*dy))
            relative.append((s["o"]-t["o"]) % 4)
        if s["cap"] and s["c"] in controlled:
            param[i] = len(state)//2
            state.extend((s["n"], relative[i]))

    markers = []
    for x, y in dots:
        candidates = []
        for i, s in enumerate(shapes):
            if (s["x"] <= x+1 < s["x"]+s["w"] and
                    s["y"] <= y+1 < s["y"]+s["h"]):
                dx, dy = DIR[s["o"]]
                tx = s["ax"]+3*(s["n"]-1)*dx
                ty = s["ay"]+3*(s["n"]-1)*dy
                candidates.append(((tx-x)**2+(ty-y)**2, i))
        if candidates:
            i = min(candidates)[1]
            s = shapes[i]
            dx, dy = DIR[s["o"]]
            markers.append((i, x-s["ax"]-3*(s["n"]-1)*dx,
                            y-s["ay"]-3*(s["n"]-1)*dy))
        else:
            markers.append((-1, x, y))

    ancestors = []
    for i in range(n):
        chain, p = [], i
        while p != -1:
            chain.append(p)
            p = parent[p]
        ancestors.append(chain)

    # Screen clipping is not physical collision. Independent rotary bars
    # can swing beyond the visible area; articulated workspace boundaries
    # are kept conservative when their off-screen walls cannot be observed.
    has_children = set(p for p in parent if p >= 0)
    outside = {i for i, s in enumerate(shapes)
               if parent[i] == -1 and i not in has_children
               and s["c"] in rot_colors}

    m = dict(shapes=shapes, parent=parent, order=order,
             param=param, offsets=offsets, relative=relative,
             markers=markers, goals=goals, buttons=buttons,
             ignore=ignore, fixed=fixed, prefix=prefix(fixed),
             rot_colors=rot_colors, length_colors=length_colors,
             ancestors=ancestors, rx=rx, ry=ry, databases={},
             outside=outside)
    state = tuple(state)
    m["distances"] = goal_distances(m)
    m["allowed"] = allowed_assignments(m, state)
    return m, state


def relaxed_transform(table, lengths, angles, m):
    dims = table.ndim
    for axis in range(len(lengths)):
        for j in range(1, 21):
            dst = [slice(None)]*dims
            src = dst[:]
            dst[axis], src[axis] = j, j-1
            d, s = tuple(dst), tuple(src)
            table[d] = np.minimum(table[d], table[s]+1)
        for j in range(19, -1, -1):
            dst = [slice(None)]*dims
            src = dst[:]
            dst[axis], src[axis] = j, j+1
            d, s = tuple(dst), tuple(src)
            table[d] = np.minimum(table[d], table[s]+1)
    for offset, (_, i) in enumerate(angles):
        axis = len(lengths)+offset
        successors = ((3, 0, 1, 1) if m["parent"][i] != -1
                      else (3, 0, 1, 2))
        for _ in range(4):
            for j, nxt in enumerate(successors):
                dst = [slice(None)]*dims
                src = dst[:]
                dst[axis], src[axis] = j, nxt
                d, s = tuple(dst), tuple(src)
                table[d] = np.minimum(table[d], table[s]+1)


def advance_reverse(table, progress, deadline):
    if progress["done"]:
        return
    flat = table.reshape(-1)
    frontier = progress["frontier"]
    depth = progress["depth"]
    dims = progress["dims"]
    children = progress["children"]
    strides = progress["strides"]
    nlength = len(dims)-len(children)

    while frontier.size and depth < INF-2:
        if time.monotonic() > deadline:
            break
        candidates = []
        for axis, (dimension, stride) in enumerate(zip(dims, strides)):
            coordinate = (frontier//stride) % dimension
            if axis < nlength:
                for a in (frontier[coordinate > 0]-stride,
                          frontier[coordinate < dimension-1]+stride):
                    if a.size:
                        a = a[flat[a] == INF]
                        if a.size:
                            candidates.append(a)
            else:
                child = children[axis-nlength]
                successors = (3, 0, 1, 1) if child else (3, 0, 1, 2)
                for previous, destination in enumerate(successors):
                    selected = frontier[coordinate == destination]
                    if selected.size:
                        a = selected+(previous-destination)*stride
                        a = a[flat[a] == INF]
                        if a.size:
                            candidates.append(a)
        if not candidates:
            frontier = np.empty(0, dtype=np.int32)
            break
        frontier = np.unique(np.concatenate(candidates))
        depth += 1
        flat[frontier] = depth

    progress["frontier"] = frontier
    progress["depth"] = depth
    progress["done"] = not frontier.size or depth >= INF-2


def build_database(m, initial, mi, gj, deadline):
    p = m["markers"][mi][0]
    chain = m["ancestors"][p][::-1] if p >= 0 else []
    lengths, angles = [], []
    for i in chain:
        q = m["param"][i]
        if q < 0:
            continue
        color = m["shapes"][i]["c"]
        if color in m["length_colors"]:
            lengths.append((q, i))
        if color in m["rot_colors"]:
            angles.append((q, i))

    dims = [21]*len(lengths)+[4]*len(angles)
    size = math.prod(dims)
    if size > 14000000 or len(lengths) > 4:
        return None

    scope = tuple([2*q for q, _ in lengths] +
                  [2*q+1 for q, _ in angles])
    if not dims:
        _, marks = geometry(m, initial)
        value = 0 if marks[mi] == m["goals"][gj] else INF
        return scope, np.array(value, dtype=np.uint8), None, None

    lshape = (21,)*len(lengths)
    ranges = []
    for k in range(len(lengths)):
        shape = [1]*len(lengths)
        shape[k] = 21
        ranges.append(np.arange(21, dtype=np.int16).reshape(shape))

    legal_table = np.full(tuple(dims), 255, dtype=np.uint8)
    relaxed = np.full(tuple(dims), INF, dtype=np.uint8)
    pp = np.asarray(m["prefix"], dtype=np.int32)
    goal = m["goals"][gj]
    length_map = {i: k for k, (_, i) in enumerate(lengths)}

    for av in itertools.product(range(4), repeat=len(angles)):
        if time.monotonic() > deadline:
            return None
        base = list(initial)
        for q, _ in lengths:
            base[2*q] = 1
        for (q, _), value in zip(angles, av):
            base[2*q+1] = value
        geo, marks = geometry(m, base)

        legal = np.ones(lshape, dtype=bool)
        rectangles = []
        for i in chain:
            ax, ay, ln, orient, _ = geo[i]
            xs = np.full(lshape, ax, dtype=np.int16)
            ys = np.full(lshape, ay, dtype=np.int16)
            for q, a in lengths:
                if a != i and a in m["ancestors"][i]:
                    r = ranges[length_map[a]]
                    dx, dy = DIR[geo[a][3]]
                    if dx:
                        xs += 3*dx*r
                    if dy:
                        ys += 3*dy*r
            extent = ranges[length_map[i]] if i in length_map else ln-1
            dx, dy = DIR[orient]
            xe, ye = xs+3*dx*extent, ys+3*dy*extent
            x1, y1 = np.minimum(xs, xe), np.minimum(ys, ye)
            x2, y2 = np.maximum(xs, xe)+3, np.maximum(ys, ye)+3
            if i not in m["outside"]:
                legal &= (x1 >= 0) & (y1 >= 0) & (x2 <= 64) & (y2 <= 64)
            cx1, cy1 = np.clip(x1, 0, 64), np.clip(y1, 0, 64)
            cx2, cy2 = np.clip(x2, 0, 64), np.clip(y2, 0, 64)
            hits = pp[cy2, cx2]-pp[cy1, cx2]-pp[cy2, cx1]+pp[cy1, cx1]
            legal &= hits == 0
            for bx1, by1, bx2, by2 in rectangles:
                legal &= ~((x1 < bx2) & (bx1 < x2) &
                           (y1 < by2) & (by1 < y2))
            rectangles.append((x1, y1, x2, y2))

        mx = np.full(lshape, marks[mi][0], dtype=np.int16)
        my = np.full(lshape, marks[mi][1], dtype=np.int16)
        for q, i in lengths:
            dx, dy = DIR[geo[i][3]]
            r = ranges[length_map[i]]
            if dx:
                mx += 3*dx*r
            if dy:
                my += 3*dy*r
        hits = legal & (mx == goal[0]) & (my == goal[1])
        index = (slice(None),)*len(lengths)+av
        legal_table[index] = np.where(hits, 0,
                                     np.where(legal, INF, 255)).astype(np.uint8)
        relaxed[index] = np.where(hits, 0, INF).astype(np.uint8)

    relaxed_transform(relaxed, lengths, angles, m)
    strides, product = [], 1
    for d in dims[::-1]:
        strides.append(product)
        product *= d
    strides.reverse()
    progress = dict(
        dims=dims, strides=strides,
        children=[m["parent"][i] != -1 for _, i in angles],
        frontier=np.flatnonzero(legal_table.reshape(-1) == 0).astype(np.int32),
        depth=0, done=False)
    advance_reverse(legal_table, progress, deadline)
    return scope, relaxed, legal_table, progress


def build_tables(m, state, deadline):
    _, marks = geometry(m, state)
    pairs = []
    for i in range(len(marks)):
        for j, goal in enumerate(m["goals"]):
            if m["allowed"][i][j]:
                p = m["markers"][i][0]
                count = len(m["ancestors"][p]) if p >= 0 else 0
                pairs.append((marks[i] == goal, -count, i, j))
    total = 0
    for _, _, i, j in sorted(pairs):
        if time.monotonic() > deadline:
            break
        db = build_database(m, state, i, j, deadline)
        if db is not None:
            m["databases"][i, j] = db
            total += db[1].size
            if db[2] is not None:
                total += db[2].size
        if total > 65000000:
            break


def continue_tables(m, deadline):
    for scope, relaxed, exact, progress in m["databases"].values():
        if time.monotonic() >= deadline:
            return
        if exact is not None and progress is not None and not progress["done"]:
            advance_reverse(exact, progress, deadline)


def clearance(m, geo):
    required = []
    for p, vx, vy in m["markers"]:
        if p < 0 or vx or vy or m["parent"][p] != -1:
            continue
        s = m["shapes"][p]
        if s["c"] in m["rot_colors"]:
            continue
        ax, ay, n, o, _ = geo[p]
        dx, dy = DIR[o]
        choices = []
        for gx, gy in m["goals"]:
            xx, yy = gx-ax, gy-ay
            distance = xx*dx+yy*dy
            if xx*dy == yy*dx and distance >= 0 and distance % 3 == 0:
                choices.append((abs(distance//3+1-n), distance//3+1))
        if choices:
            required.append((p, box(ax, ay, min(choices)[1], o)))

    needs = {}
    for owner, target in required:
        for b, g in enumerate(geo):
            if b == owner or owner in m["ancestors"][b]:
                continue
            if not overlap(g[4], target):
                continue
            best = None
            for a in m["ancestors"][b]:
                s = m["shapes"][a]
                color = s["c"]
                if m["param"][a] < 0:
                    continue
                if color in m["rot_colors"]:
                    candidate = (1, color)
                    if best is None or candidate < best:
                        best = candidate
                if color not in m["length_colors"]:
                    continue
                ax, ay, ln, o, _ = geo[a]
                if a == b:
                    for newlen in range(ln-1, 0, -1):
                        if not overlap(box(ax, ay, newlen, o), target):
                            candidate = (ln-newlen, color)
                            if best is None or candidate < best:
                                best = candidate
                            break
                else:
                    dx, dy = DIR[o]
                    bx, by, bw, bh = g[4]
                    tx, ty, tw, th = target
                    if dx:
                        plus = (tx+tw-bx+2)//3
                        minus = (bx+bw-tx+2)//3
                        if dx < 0:
                            plus, minus = minus, plus
                    else:
                        plus = (ty+th-by+2)//3
                        minus = (by+bh-ty+2)//3
                        if dy < 0:
                            plus, minus = minus, plus
                    choices = []
                    if plus > 0 and ln+plus <= 21:
                        choices.append(plus)
                    if minus > 0 and ln-minus >= 1:
                        choices.append(minus)
                    if choices:
                        candidate = (min(choices), color)
                        if best is None or candidate < best:
                            best = candidate
            if best:
                value, color = best
                needs[color] = max(needs.get(color, 0), value)
    return sum(needs.values())


def heuristic(m, state, geo, marks):
    ng, nm = len(m["goals"]), len(marks)
    if not ng:
        return 0.0
    if nm < ng:
        return 1000.0
    costs = []
    for j, goal in enumerate(m["goals"]):
        row = []
        for i, pos in enumerate(marks):
            if not m["allowed"][i][j]:
                row.append(1000.0)
                continue
            db = m["databases"].get((i, j))
            if db is None:
                man = (abs(pos[0]-goal[0])+abs(pos[1]-goal[1]))/3.0
                value = m["distances"][j].get(pos, man+12)
            else:
                scope, relaxed, exact, progress = db
                index = tuple(state[k] if k % 2 else state[k]-1
                              for k in scope)
                value = int(relaxed[index])
                if exact is not None:
                    ev = int(exact[index])
                    if ev < INF:
                        value = max(value, ev)
                    elif progress["done"]:
                        value = 1000
                if value >= INF:
                    value = 1000
            row.append(float(value))
        costs.append(row)

    dp = {0: 0.0}
    for row in costs:
        nd = {}
        for mask, value in dp.items():
            for i, cost in enumerate(row):
                if not ((mask >> i) & 1):
                    key = mask | (1 << i)
                    nv = value+cost
                    if nv < nd.get(key, float("inf")):
                        nd[key] = nv
        dp = nd
    return (min(dp.values()) if dp else 1000.0)+clearance(m, geo)


def solved(m, marks):
    return bool(m["goals"]) and set(m["goals"]).issubset(set(marks))


def path_to(parents, state):
    path = []
    while state in parents:
        previous, action = parents[state]
        path.append(action)
        state = previous
    return path[::-1]


def search(m, start, forbidden, visited, deadline):
    geo, marks = geometry(m, start)
    if solved(m, marks):
        return []
    h = heuristic(m, start, geo, marks)
    heap = [(1.7*h, h, 0, 0, start)]
    costs = {start: 0}
    parents = {}
    serial = 0
    best, best_score = None, float("inf")
    forbidden = set(forbidden)
    expanded = 0
    actions = range(len(m["buttons"]))

    while heap and expanded < 160000:
        if expanded % 32 == 0 and time.monotonic() > deadline:
            break
        _, h, g, _, state = heapq.heappop(heap)
        if costs.get(state) != g:
            continue
        expanded += 1
        if g:
            score = h + 0.06*g + 5*visited.get(state, 0)
            if score < best_score:
                best_score, best = score, state
        for action in actions:
            if state == start and action in forbidden:
                continue
            nxt = transition(m, state, action)
            if nxt is None or costs.get(nxt, 10**9) <= g+1:
                continue
            ng, nm = geometry(m, nxt)
            if not valid(m, ng):
                continue
            costs[nxt] = g+1
            parents[nxt] = (state, action)
            if solved(m, nm):
                return path_to(parents, nxt)
            nh = heuristic(m, nxt, ng, nm)
            serial += 1
            heapq.heappush(heap, (g+1+1.7*nh, nh, g+1, serial, nxt))

    if best is not None:
        return path_to(parents, best)
    candidates = []
    for action in actions:
        if action in forbidden:
            continue
        nxt = transition(m, start, action)
        if nxt is None:
            continue
        ng, nm = geometry(m, nxt)
        if valid(m, ng):
            candidates.append((heuristic(m, nxt, ng, nm) +
                               5*visited.get(nxt, 0), action))
    return [min(candidates)[1]] if candidates else []


def prediction(m, state):
    frame = np.full((64, 64), 5, dtype=np.int8)
    frame[m["fixed"]] = 15
    geo, _ = geometry(m, state)
    for s, g in zip(m["shapes"], geo):
        _, _, _, o, (x, y, w, h) = g
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(64, x+w), min(64, y+h)
        if x1 >= x2 or y1 >= y2:
            continue
        frame[y1:y2, x1:x2] = s["c"]
        if s["cap"]:
            if o == 0:
                yy = y+h-1
                if 0 <= yy < 64:
                    frame[yy, x1:x2] = 3
            elif o == 1:
                if 0 <= x < 64:
                    frame[y1:y2, x] = 3
            elif o == 2:
                if 0 <= y < 64:
                    frame[y, x1:x2] = 3
            else:
                xx = x+w-1
                if 0 <= xx < 64:
                    frame[y1:y2, xx] = 3
    return frame


def act(obs, available, memory):
    deadline = time.monotonic()+8.8
    obs = np.asarray(obs, dtype=np.int8)
    names = [str(a).upper() for a in available]
    if "ACTION6" not in names:
        return ("RESET" if "RESET" in names else names[0]), None, None

    signature = obs[:63].tobytes()
    bans = memory.setdefault("bans", {})
    matching, unchanged = False, False
    if "model" in memory and "expected" in memory:
        m = memory["model"]
        check = ~m["ignore"] & (obs != 13)
        matching = bool(np.all(obs[check] == memory["expected"][check]))
        unchanged = signature == memory.get("previous_signature")
        if not matching and unchanged:
            click = memory.get("previous_click")
            if click is not None:
                bans.setdefault(signature, set()).add(click)

    fresh_model = False
    if matching:
        m = memory["model"]
        state = memory["state"]
        plan = memory.get("plan", [])
        visited = memory.setdefault("visited", {})
    elif unchanged and "previous_state" in memory:
        m = memory["model"]
        state = memory["previous_state"]
        memory["state"] = state
        plan = []
        visited = memory.setdefault("visited", {})
    else:
        m, state = make_model(obs)
        build_tables(m, state, deadline-1.8)
        memory["model"] = m
        memory["state"] = state
        memory["visited"] = {}
        visited = memory["visited"]
        plan = []
        fresh_model = True

    if not fresh_model:
        allowance = 6.0 if plan else 1.5
        continue_tables(m, min(deadline-0.3,
                               time.monotonic()+allowance))

    visited[state] = visited.get(state, 0)+1
    if not plan:
        bad = bans.get(signature, set())
        forbidden = [i for i, b in enumerate(m["buttons"])
                     if (b[2], b[3]) in bad]
        plan = search(m, state, forbidden, visited, deadline)

    if plan:
        action = plan[0]
        nxt = transition(m, state, action)
        if nxt is not None:
            b = m["buttons"][action]
            memory["previous_state"] = state
            memory["state"] = nxt
            memory["expected"] = prediction(m, nxt)
            memory["plan"] = plan[1:]
            memory["previous_signature"] = signature
            memory["previous_click"] = (int(b[2]), int(b[3]))
            return "ACTION6", int(b[2]), int(b[3])

    memory.pop("expected", None)
    memory["plan"] = []
    if "RESET" in names and not memory.get("reset_used"):
        memory["reset_used"] = True
        memory.pop("model", None)
        memory.pop("previous_state", None)
        return "RESET", None, None

    bad = bans.get(signature, set())
    for b in m["buttons"]:
        if (b[2], b[3]) not in bad:
            memory["previous_state"] = state
            memory["previous_signature"] = signature
            memory["previous_click"] = (int(b[2]), int(b[3]))
            memory["expected"] = prediction(m, state)
            return "ACTION6", int(b[2]), int(b[3])
    return "ACTION6", 0, 0

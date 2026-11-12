import numpy as np
import heapq
import time
from collections import deque

_W = 128
_DS = (-128, 128, -1, 1)
_AN = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
_CIRCLE = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=bool)


def _p(x, y):
    return (y + 32) * _W + x + 32


def _xy(p):
    return p % _W - 32, p // _W - 32


def _screen(p, off):
    x, y = _xy(p)
    return off[0] + 6*x + 3, off[1] + 6*y + 3


def _visible(p, off):
    x, y = _screen(p, off)
    return 0 <= x < 64 and 1 <= y < 64


def _count(s):
    return len(s[0]) + len(s[1])


def _viewkey(s, off):
    return tuple(tuple(p for p in g if _visible(p, off)) for g in s)


def _phase(obs, old):
    best, score0 = (0, 0), -1e30
    for oy in range(6):
        for ox in range(6):
            h, w = (64-oy)//6, (64-ox)//6
            a = obs[oy:oy+6*h, ox:ox+6*w]
            a = a.reshape(h, 6, w, 6).transpose(0, 2, 1, 3)
            co = np.stack((a[:, :, 0, 0], a[:, :, 0, 5],
                           a[:, :, 5, 0], a[:, :, 5, 5]), axis=-1)
            inn = a[:, :, 1:5, 1:5]
            floor = ((np.sum(co == 0, axis=-1) == 4)
                     & (np.sum(inn == 10, axis=(-1, -2)) == 0)
                     & (np.sum(inn != 0, axis=(-1, -2)) >= 5))
            plat = np.sum(co == 11, axis=-1) >= 3
            score = 30*float(np.sum(floor)) + 34*float(np.sum(plat))
            score += .25*float(np.sum((inn == 1)*floor[:, :, None, None]))
            if old is not None:
                score += .1*(ox == old[0] % 6) + .1*(oy == old[1] % 6)
            if score > score0:
                score0, best = score, (ox, oy)
    return best


def _read(obs, old):
    ox, oy = _phase(obs, old)
    cells = {}
    gray = 0
    for gy in range(-1, 11):
        sy = oy + 6*gy
        for gx in range(-1, 11):
            sx = ox + 6*gx
            x0, x1 = max(0, sx), min(64, sx+6)
            y0, y1 = max(1, sy), min(64, sy+6)
            if x0 >= x1 or y0 >= y1:
                continue
            a = np.full((6, 6), -1, dtype=np.int16)
            a[y0-sy:y1-sy, x0-sx:x1-sx] = obs[y0:y1, x0:x1]
            c = a[2:4, 2:4]
            c = c[c >= 0]
            if len(c) < 2:
                continue
            corners = np.array((a[0, 0], a[0, 5], a[5, 0], a[5, 5]))
            corners = corners[corners >= 0]
            inner = a[1:5, 1:5]
            inner = inner[inner >= 0]
            floor = (len(corners) >= 2 and np.all(corners == 0)
                     and len(inner) >= 4 and not np.any(inner == 10)
                     and np.count_nonzero(inner != 0) >= 3)
            platform = (len(corners) >= 2
                        and np.count_nonzero(corners == 11) >= len(corners)-1
                        and np.count_nonzero(a == 11) >= 3)
            rail = not floor and not platform and np.all(c == 5)
            terrain = 1 if floor else 2 if platform or rail else 0
            peg, post = -1, False
            if floor or platform:
                post = (np.count_nonzero(a == 15) >= 3
                        or np.count_nonzero(a == 7) >= 2)
                if not post:
                    for color, kind in ((14, 0), (8, 1), (9, 2),
                                        (2, 17), (4, 0)):
                        if np.count_nonzero(c == color) < min(2, len(c)):
                            continue
                        mask = _CIRCLE & (a >= 0)
                        n = int(np.count_nonzero(mask))
                        hits = int(np.count_nonzero((a == color) & mask))
                        if n >= 4 and hits >= max(4, int(.72*n)):
                            peg = kind
                            if kind == 17:
                                gray += 1
                            break
            full = x1-x0 == 6 and y1-y0 == 6
            cells[_p(gx, gy)] = (terrain, peg, post, platform, full)
    return (ox, oy), cells, gray


def _align(phase, cells, m):
    old = m["off"]
    terrain = m["terrain"]
    s = m["s"]
    pegs = {p: c for c in range(3) for p in s[c]}
    posts, pads = set(s[4]), set(s[3])
    cx = int(round((phase[0]-old[0])/6))
    cy = int(round((phase[1]-old[1])/6))
    best, bestscore = (cx, cy), -1e30
    for dy in range(cy-11, cy+12):
        for dx in range(cx-11, cx+12):
            shift = dx + _W*dy
            off = (phase[0]-6*dx, phase[1]-6*dy)
            score = -.055*(abs(off[0]-old[0]) + abs(off[1]-old[1]))
            for p, (t, peg, post, pad, full) in cells.items():
                q = p + shift
                known = terrain.get(q)
                if known is not None:
                    if known == t:
                        score += 5 if t == 1 else 3 if t == 2 else .12
                    else:
                        score -= 6 if known == 1 or t == 1 else 2.6
                ep = pegs.get(q, -1)
                if peg >= 0:
                    score += 10 if ep == peg or peg == 17 and ep >= 0 else -2
                elif ep >= 0:
                    score -= 3
                if post:
                    score += 7 if q in posts else -1
                elif q in posts:
                    score -= 2
                if pad:
                    score += 7 if q in pads else -1
                elif q in pads:
                    score -= 2
            if score > bestscore:
                bestscore, best = score, (dx, dy)
    dx, dy = best
    return (phase[0]-6*dx, phase[1]-6*dy), dx + _W*dy


def _init(m):
    m.update(
        terrain={}, s=((), (), (), (), ()), off=(0, 0),
        plan=[], journal=[], bad=set(), visits={}, reached=set(),
        pan={}, jump_pan={}, traps={}, unrestricted=False,
        viewbad=set(), colored_seen=set()
    )


def _observe(obs, m):
    fresh = "s" not in m
    phase, cells, gray = _read(obs, m.get("off"))
    if fresh:
        _init(m)
        off, shift = phase, 0
        m["off"] = phase
        m["initial_off"] = phase
    else:
        off, shift = _align(phase, cells, m)

    old_s, old_off = m["s"], m["off"]
    old_pegs = {p: c for c in range(3) for p in old_s[c]}
    seen = {p+shift for p in cells}
    groups = [set(g)-seen for g in old_s]
    discovery = False

    for p, (t, peg, post, pad, full) in cells.items():
        q = p + shift
        known = m["terrain"].get(q)
        if t and known != t:
            discovery = True
        if t or known is None or known == 0 or (full and known == 2):
            if known != t and known is not None:
                m["plan"] = []
            m["terrain"][q] = t
        if peg == 17:
            peg = old_pegs.get(q, 0)
        if 0 <= peg < 3:
            groups[peg].add(q)
            if q not in old_pegs:
                discovery = True
        if post:
            groups[4].add(q)
            if q not in old_s[4]:
                discovery = True
        if pad:
            groups[3].add(q)

    s = tuple(tuple(sorted(g)) for g in groups)
    issued = m.pop("issued", None)
    unsafe_pan = False

    if issued is not None:
        before, before_off, cmd, predicted, predicted_off = issued
        delta = (off[0]-before_off[0], off[1]-before_off[1])
        if cmd[0] == "m" and before[0] != predicted[0]:
            direction = cmd[1]
            m["pan"][direction] = delta
            if delta != (0, 0):
                opposite = (1, 0, 3, 2)[direction]
                m["pan"].setdefault(opposite, (-delta[0], -delta[1]))
                if delta[0] > 0 and before_off[0] >= m["initial_off"][0]:
                    m["unrestricted"] = True
        elif cmd[0] == "j" and delta != (0, 0):
            m["jump_pan"][(cmd[3], before_off)] = delta
            for p in s[0] + s[1]:
                if _visible(p, before_off) and not _visible(p, off):
                    unsafe_pan = True

        if s != predicted or off != predicted_off or discovery:
            m["plan"] = []
        if s == before:
            m["bad"].add((before, cmd))
        if (off == before_off
                and _viewkey(s, off) == _viewkey(before, off)
                and _viewkey(predicted, off) != _viewkey(before, off)):
            m["viewbad"].add((_viewkey(before, off), cmd))
            m["plan"] = []
        if cmd[0] == "j":
            m["colored_seen"].add((s[:3], off))

    elif discovery:
        m["plan"] = []

    m["s"], m["off"] = s, off
    if fresh or issued is not None or s != old_s or off != old_off:
        key = (s, off)
        m["visits"][key] = m["visits"].get(key, 0) + 1
    return fresh, gray, unsafe_pan, issued


def _replace(group, moves):
    return tuple(sorted(moves.get(p, p) for p in group))


def _transition(s, cmd, floor, rails):
    if cmd[0] == "j":
        _, color, src, dst = cmd
        if src not in s[color]:
            return None
        diff = dst-src
        if diff not in (-256, 256, -2, 2):
            return None
        mid = src + diff//2
        occ = set(s[0]) | set(s[1]) | set(s[2]) | set(s[4])
        if mid not in occ or dst in occ:
            return None
        if dst not in floor and dst not in s[3]:
            return None
        groups = list(s)
        own = set(groups[color])
        own.remove(src)
        own.add(dst)
        if color != 2 and mid in own:
            own.remove(mid)
        groups[color] = tuple(sorted(own))
        return tuple(groups)

    d = _DS[cmd[1]]
    pads = set(s[3])
    if not pads:
        return None
    if d == 1:
        order = sorted(pads, key=lambda p: p % _W, reverse=True)
    elif d == -1:
        order = sorted(pads, key=lambda p: p % _W)
    else:
        order = sorted(pads, reverse=d > 0)
    moves = {}
    for p in order:
        q = p+d
        if q in rails and q not in pads:
            pads.remove(p)
            pads.add(q)
            moves[p] = q
    if not moves:
        return None
    return (_replace(s[0], moves), _replace(s[1], moves),
            _replace(s[2], moves), tuple(sorted(pads)),
            _replace(s[4], moves))


def _next_off(s, ns, off, cmd, m):
    if cmd[0] == "j":
        delta = m["jump_pan"].get((cmd[3], off), (0, 0))
    elif s[0] != ns[0]:
        delta = m["pan"].get(cmd[1], (0, 0))
        if (delta[0] > 0 and off[0] >= m["initial_off"][0]
                and not m["unrestricted"]):
            delta = (0, delta[1])
    else:
        delta = (0, 0)
    return off[0]+delta[0], off[1]+delta[1]


def _legal_edge(s, ns, off, cmd, m):
    if (s, cmd) in m["bad"]:
        return False
    if m["viewbad"] and (_viewkey(s, off), cmd) in m["viewbad"]:
        return False
    if cmd[0] == "j":
        if not (_visible(cmd[2], off) and _visible(cmd[3], off)):
            return False
        if _count(ns) <= m["traps"].get(cmd[3], -1):
            return False
        noff = _next_off(s, ns, off, cmd, m)
        if noff != off:
            for p in ns[0] + ns[1]:
                if _visible(p, off) and not _visible(p, noff):
                    return False
    return True


def _jumps(s, floor):
    occ = set(s[0]) | set(s[1]) | set(s[2]) | set(s[4])
    landing = floor | set(s[3])
    for color in range(3):
        own = set(s[color])
        for src in s[color]:
            for d in _DS:
                mid, dst = src+d, src+2*d
                if mid not in occ or dst not in landing or dst in occ:
                    continue
                group = own.copy()
                group.remove(src)
                group.add(dst)
                if color != 2 and mid in own:
                    group.remove(mid)
                ns = list(s)
                ns[color] = tuple(sorted(group))
                yield tuple(ns), ("j", color, src, dst), 2.0


def _neighbors(s, floor, rails):
    yield from _jumps(s, floor)
    for direction in range(4):
        cmd = ("m", direction)
        ns = _transition(s, cmd, floor, rails)
        if ns is not None:
            yield ns, cmd, 1.0


def _geometry(floor, rails, goals):
    land = floor | rails
    graph = {}
    for p in land:
        adj = []
        for d in _DS:
            if p+d in land and p+2*d in land:
                adj.append(p+2*d)
            if p in rails and p+d in rails:
                adj.append(p+d)
        graph[p] = adj
    distances = {}
    for p in land:
        ds = {p: 0}
        queue = deque([p])
        while queue:
            q = queue.popleft()
            nd = ds[q]+1
            for r in graph[q]:
                if r not in ds:
                    ds[r] = nd
                    queue.append(r)
        distances[p] = ds

    attack = {}
    adjacent = {q: [q+d for d in _DS if q+d in land and q-d in land]
                for q in land}
    for p in land:
        ds = distances[p]
        attack[p] = {
            q: min((ds.get(v, 24) for v in adjacent[q]), default=24)
            for q in land
        }
    for p in land:
        for q in land:
            z = min(attack[p][q], attack[q][p], 18)
            attack[p][q] = attack[q][p] = z
    fd = {
        p: min((distances[p].get(g, 30) for g in goals), default=0)
        for p in land
    }
    rd = {}
    for p in rails:
        ds = {p: 0}
        queue = deque([p])
        while queue:
            q = queue.popleft()
            for d in _DS:
                r = q+d
                if r in rails and r not in ds:
                    ds[r] = ds[q]+1
                    queue.append(r)
        rd[p] = ds
    return distances, attack, fd, rd


def _path(nodes, index):
    path = []
    while index:
        key, g, parent, cmd = nodes[index]
        path.append(cmd)
        index = parent
    path.reverse()
    return path


def _blue_macros(root, off, floor, rails, potential, deadline):
    if not root[2]:
        return []
    static = set(root[0]) | set(root[1]) | set(root[4])
    land = floor | set(root[3])
    wishes = []
    for color in range(2):
        for src in root[color]:
            if not _visible(src, off):
                continue
            base = potential(src, color)
            for d in _DS:
                mid, dst = src+d, src+2*d
                if (mid in static or dst in static or dst not in land
                        or mid not in land or not _visible(dst, off)):
                    continue
                improvement = base-potential(dst, color)
                if improvement >= -.01:
                    wishes.append((color, src, mid, dst, improvement))
    if not wishes:
        return []

    start = root[2]
    queue = deque([start])
    parent = {start: (None, None, 0)}
    answers = {}
    expanded = 0
    while queue:
        blue = queue.popleft()
        prev, action, depth = parent[blue]
        bs = set(blue)
        for i, (c, src, mid, dst, improvement) in enumerate(wishes):
            if i not in answers and mid in bs and dst not in bs:
                path = []
                k = blue
                while parent[k][0] is not None:
                    pk, ac, dep = parent[k]
                    path.append(ac)
                    k = pk
                path.reverse()
                path.append(("j", c, src, dst))
                if len(path) > 1:
                    answers[i] = path
        expanded += 1
        if expanded % 128 == 0:
            if time.perf_counter() >= deadline or len(parent) >= 26000:
                break
        if len(answers) == len(wishes) or depth >= 16:
            continue
        occ = static | bs
        for src in blue:
            if not _visible(src, off):
                continue
            for d in _DS:
                mid, dst = src+d, src+2*d
                if (mid not in occ or dst in occ or dst not in land
                        or not _visible(dst, off)):
                    continue
                nb = tuple(sorted((bs-{src}) | {dst}))
                if nb not in parent:
                    parent[nb] = (blue, ("j", 2, src, dst), depth+1)
                    queue.append(nb)
    return list(answers.values())


def _plan(m):
    started = time.perf_counter()
    deadline = started + 6.8
    root, off = m["s"], m["off"]
    floor = {p for p, t in m["terrain"].items() if t == 1}
    rails = {p for p, t in m["terrain"].items() if t == 2}
    land = floor | rails
    if not land:
        return []
    target = max(1, int(bool(root[0])) + int(bool(root[1])))
    initial_n = _count(root)
    exploring = initial_n <= target
    frontier = {p for p in land
                if any(p+d not in m["terrain"] for d in _DS)}
    goals = set(frontier)
    for _ in range(2):
        goals |= {p+d for p in list(goals) for d in _DS if p+d in land}
    if exploring:
        goals |= {p for p in land
                  if 49 <= _screen(p, off)[0] < 64}
    root_pegs = set(root[0]) | set(root[1])
    goals = {
        p for p in goals
        if p not in root_pegs and (off, p) not in m["reached"]
        and _visible(p, off)
    }
    distances, attack, fd, rd = _geometry(floor, rails, goals)

    def potential(p, color):
        others = [q for q in root[color] if q != p]
        if len(root[color]) > 1:
            return min(attack.get(p, {}).get(q, 18) for q in others) \
                if others else 0
        return fd.get(p, 30) if goals else 0

    def heuristic(s, viewport):
        n = _count(s)
        h = 6.5*max(0, n-target)
        for color in range(2):
            group = s[color]
            if len(group) > 1:
                for p in group:
                    ap = attack.get(p, {})
                    h += .70*min(ap.get(q, 18) for q in group if q != p)
        if goals:
            vals = [fd.get(p, 30) for group in s[:2] for p in group]
            if vals:
                h += (.85 if exploring else .30)*min(vals)
                h += .045*sum(vals)

        occ = set(s[0]) | set(s[1]) | set(s[2]) | set(s[4])
        pads = set(s[3])
        landing = floor | pads
        if pads and n <= 5:
            required = []
            for group in s[:2]:
                for p in group:
                    if p in pads:
                        continue
                    for d in _DS:
                        if p+d in occ and p+2*d in rails and p+2*d not in occ:
                            required.append(p+2*d)
            if required:
                best = min(
                    (rd.get(p, {}).get(q, 20) for p in pads for q in required),
                    default=20
                )
                h += .45*best

        if s[2] and n <= 4:
            usable = [b for b in s[2] if _visible(b, viewport)]
            if usable:
                for color in range(2):
                    for p in s[color]:
                        if p in pads or not _visible(p, viewport):
                            continue
                        legal = False
                        wishes = []
                        for d in _DS:
                            mid, dst = p+d, p+2*d
                            if dst not in landing or dst in occ:
                                continue
                            if mid in occ:
                                legal = True
                                break
                            if mid in land:
                                if len(s[color]) > 1:
                                    other = [q for q in s[color] if q != p]
                                    merit = min(attack.get(dst, {}).get(q, 18)
                                                for q in other)
                                else:
                                    merit = fd.get(dst, 20)
                                wishes.append((merit, mid))
                        if not legal and wishes:
                            minimum = min(a for a, b in wishes)
                            values = []
                            for merit, mid in wishes:
                                if merit > minimum+1:
                                    continue
                                direct = min(distances.get(b, {}).get(mid, 24)
                                             for b in usable)
                                support = min(
                                    (distances.get(b, {}).get(mid+d, 24)
                                     for b in usable for d in _DS),
                                    default=24
                                )
                                values.append(.48*direct + .12*support)
                            h += min(10, 1.5+min(values, default=10))
        return h

    rootkey = (root, off)
    h0 = heuristic(root, off)
    nodes = [(rootkey, 0.0, -1, None)]
    heap = [(2.0*h0, 0, 0)]
    costs = {rootkey: 0.0}
    serial = 1
    best_index = 0
    best_quality = h0
    frontier_index = None
    frontier_quality = 1e30

    if root[2]:
        macros = _blue_macros(
            root, off, floor, rails, potential,
            min(deadline-.8, time.perf_counter()+1.15)
        )
        for path in macros:
            s, viewport, parent, g = root, off, 0, 0.0
            valid = True
            for cmd in path:
                ns = _transition(s, cmd, floor, rails)
                if ns is None or not _legal_edge(s, ns, viewport, cmd, m):
                    valid = False
                    break
                noff = _next_off(s, ns, viewport, cmd, m)
                g += 2.0
                key = (ns, noff)
                ni = len(nodes)
                nodes.append((key, g, parent, cmd))
                s, viewport, parent = ns, noff, ni
            if valid and g < costs.get((s, viewport), 1e30):
                costs[(s, viewport)] = g
                nh = heuristic(s, viewport)
                heapq.heappush(heap, (g+2*nh, serial, parent))
                serial += 1

    expanded = 0
    while heap:
        if expanded % 96 == 0:
            if time.perf_counter() >= deadline or len(nodes) >= 130000:
                break
        _, _, index = heapq.heappop(heap)
        (s, viewport), g, parent, cmd = nodes[index]
        if g > costs.get((s, viewport), 1e30)+1e-8:
            continue
        expanded += 1

        if index:
            n = _count(s)
            visited = m["visits"].get((s, viewport), 0)
            if (initial_n > target and n <= target
                    and cmd[0] == "j" and not visited):
                return _path(nodes, index)

            reached = any(p in goals for group in s[:2] for p in group)
            if reached and not visited:
                fq = g + 7.0*max(0, n-target)
                if fq < frontier_quality:
                    frontier_quality, frontier_index = fq, index
                if exploring:
                    return _path(nodes, index)

            h = heuristic(s, viewport)
            quality = h + .026*g + min(8, 2.3*visited)
            if quality < best_quality-.10:
                best_quality, best_index = quality, index

        for ns, move, cost in _neighbors(s, floor, rails):
            if not _legal_edge(s, ns, viewport, move, m):
                continue
            noff = _next_off(s, ns, viewport, move, m)
            key = (ns, noff)
            ng = g+cost
            if ng >= costs.get(key, 1e30)-1e-8:
                continue
            costs[key] = ng
            ni = len(nodes)
            nodes.append((key, ng, index, move))
            nh = heuristic(ns, noff)
            heapq.heappush(heap, (ng+2.0*nh, serial, ni))
            serial += 1

    if best_index:
        return _path(nodes, best_index)
    if frontier_index is not None:
        return _path(nodes, frontier_index)
    return []


def _components(rails):
    remaining = set(rails)
    result = []
    while remaining:
        p = remaining.pop()
        component = {p}
        queue = [p]
        while queue:
            q = queue.pop()
            for d in _DS:
                r = q+d
                if r in remaining:
                    remaining.remove(r)
                    component.add(r)
                    queue.append(r)
        result.append(component)
    return result


def _rail_plan(m):
    """Search at jump boundaries, using movement-only BFS for each edge."""
    deadline = time.perf_counter() + 6.6
    fullroot, off = m["s"], m["off"]
    floor = {p for p, t in m["terrain"].items() if t == 1}
    allrails = {p for p, t in m["terrain"].items() if t == 2}
    components = _components(allrails)
    active = []
    colored = set(fullroot[0]) | set(fullroot[1])
    for comp in components:
        if any(_visible(p, off) for p in comp) or comp & colored:
            active.append(comp)
    rails = set().union(*active) if active else set()
    removedpads = set(fullroot[3])-rails
    root = (
        fullroot[0], fullroot[1], fullroot[2],
        tuple(p for p in fullroot[3] if p in rails),
        tuple(p for p in fullroot[4] if p not in removedpads)
    )
    land = floor | rails
    if not rails:
        return []
    fixedposts = set(root[4])-set(root[3])
    postrails = set()
    for comp in active:
        if comp & set(root[4]) & set(root[3]):
            postrails.update(comp)

    # Optimistic distance to a rail, ignoring the availability of jump supports.
    basegraph = {}
    for p in land:
        basegraph[p] = [
            p+2*d for d in _DS if p+d in land and p+2*d in land
        ]
        if p in rails:
            basegraph[p].extend(p+d for d in _DS if p+d in rails)
    to_rail = {p: 0 for p in rails}
    queue = deque(rails)
    while queue:
        p = queue.popleft()
        for q in basegraph[p]:
            if q not in to_rail:
                to_rail[q] = to_rail[p]+1
                queue.append(q)

    target = max(1, int(bool(root[0])) + int(bool(root[1])))
    hcache = {}

    def relaxed(s, src):
        other = (set(s[0]) | set(s[1]) | set(s[2]))-{src}
        occupied = other | fixedposts
        supports = occupied | postrails
        ds = {src: 0}
        queue = deque([src])
        while queue:
            p = queue.popleft()
            nd = ds[p]+1
            if p in rails:
                for d in _DS:
                    q = p+d
                    if q in rails and q not in occupied and q not in ds:
                        ds[q] = nd
                        queue.append(q)
            for d in _DS:
                mid, q = p+d, p+2*d
                if (mid in supports and q in land and q not in occupied
                        and q not in ds):
                    ds[q] = nd
                    queue.append(q)
        return ds

    def heuristic(s):
        key = s[:3]
        if key in hcache:
            return hcache[key]
        n = _count(s)
        if n <= target:
            return 0.0
        allpieces = s[0]+s[1]+s[2]
        reach = {p: relaxed(s, p) for p in allpieces}
        occupied = set(allpieces) | fixedposts
        best = 1000
        for color in range(2):
            for p in s[color]:
                ds = reach[p]
                for q in s[color]:
                    if p == q:
                        continue
                    for d in _DS:
                        # Move to a square from which this piece captures q.
                        start, end = q+d, q-d
                        if end in land and end not in occupied-{p}:
                            best = min(best, ds.get(start, 1000)+1)
                        # Or move where q can capture this piece.
                        victim, end = q+d, q+2*d
                        if end in land and end not in occupied-{p}:
                            best = min(best, ds.get(victim, 1000)+1)
        base = 32.0*max(0, n-target)
        if best < 1000:
            value = base + 1.45*best
        else:
            value = base + 24.0
            for color in range(2):
                if len(s[color]) < 2:
                    continue
                for p in s[color]:
                    ds = reach[p]
                    rd = min((v for q, v in ds.items() if q in rails),
                             default=1000)
                    if rd < 1000:
                        value += .55*rd
                        continue
                    helper = 1000
                    for start, dist in ds.items():
                        for d in _DS:
                            mid, end = start+d, start+2*d
                            if (mid not in land or end not in land
                                    or end in occupied-{p}
                                    or mid in fixedposts):
                                continue
                            for other in allpieces:
                                if other == p:
                                    continue
                                travel = reach[other].get(mid, 1000)
                                if travel < 1000:
                                    val = (.45*dist + .8*travel
                                           + .6*to_rail.get(end, 15) + 2)
                                    helper = min(helper, val)
                    value += 7 + min(18, helper)
        hcache[key] = value
        return value

    def reconstruct(nodes, index):
        chunks = []
        while index:
            s, g, parent, segment = nodes[index]
            chunks.append(segment)
            index = parent
        result = []
        for segment in reversed(chunks):
            result.extend(segment)
        return result

    def movement_relevant(s):
        pads = set(s[3])
        occ = set(s[0]) | set(s[1]) | set(s[2]) | fixedposts
        for p in s[0]+s[1]:
            if p in pads:
                return True
            for d in _DS:
                mid, dst = p+d, p+2*d
                if dst not in land or dst in occ:
                    continue
                if (dst in rails and (mid in occ or mid in postrails)
                        or mid in postrails):
                    return True
        return False

    def macros(s, limit):
        low = [(s, -1, None, 0)]
        seen = {s}
        head = 0
        candidates = {}
        relevant = movement_relevant(s)
        while head < len(low) and head < limit:
            st, parent, move, depth = low[head]
            if head % 64 == 0 and time.perf_counter() >= deadline:
                break
            for ns, cmd, cost in _jumps(st, floor):
                if cmd[1] == 2 or not _legal_edge(st, ns, off, cmd, m):
                    continue
                if _next_off(st, ns, off, cmd, m) != off:
                    continue
                padset = set(ns[3])
                signature = (
                    ns[:3],
                    tuple(p for p in ns[0]+ns[1] if p in padset)
                )
                score = depth+2
                variants = candidates.setdefault(signature, [])
                if any(v[0] == ns for v in variants):
                    continue
                item = (ns, score, head, cmd)
                if len(variants) < 2:
                    variants.append(item)
                elif score < variants[-1][1]:
                    variants[-1] = item
                    variants.sort(key=lambda z: z[1])
            if relevant and depth < 42:
                for direction in range(4):
                    cmd = ("m", direction)
                    ns = _transition(st, cmd, floor, rails)
                    if ns is None or ns in seen:
                        continue
                    if not _legal_edge(st, ns, off, cmd, m):
                        continue
                    if _next_off(st, ns, off, cmd, m) != off:
                        continue
                    seen.add(ns)
                    low.append((ns, head, cmd, depth+1))
            head += 1

        ranked = []
        for variants in candidates.values():
            for ns, cost, index, cmd in variants:
                h = heuristic(ns)
                penalty = 5 if (ns[:3], off) in m["colored_seen"] else 0
                ranked.append((h+.10*cost+penalty, ns, cost, index, cmd))
        ranked.sort(key=lambda z: z[0])
        result = []
        for _, ns, cost, index, cmd in ranked[:34]:
            path = [cmd]
            while index:
                state, parent, move, depth = low[index]
                path.append(move)
                index = parent
            path.reverse()
            result.append((ns, cost, path))
        return result

    h0 = heuristic(root)
    nodes = [(root, 0.0, -1, [])]
    heap = [(1.8*h0, 0, 0)]
    costs = {root: 0.0}
    serial = 1
    best_index = None
    best_quality = 1e30
    expanded = 0
    while heap and time.perf_counter() < deadline:
        _, _, index = heapq.heappop(heap)
        s, g, parent, segment = nodes[index]
        if g > costs.get(s, 1e30)+1e-8:
            continue
        if index and _count(s) <= target:
            return reconstruct(nodes, index)
        expanded += 1
        limit = 2400 if expanded < 6 else 1400
        for ns, cost, path in macros(s, limit):
            ng = g+cost
            if ng >= costs.get(ns, 1e30)-1e-8:
                continue
            costs[ns] = ng
            ni = len(nodes)
            nodes.append((ns, ng, index, path))
            h = heuristic(ns)
            seenpenalty = 6 if (ns[:3], off) in m["colored_seen"] else 0
            quality = h + .075*ng + seenpenalty
            if ns[:3] != root[:3] and quality < best_quality:
                best_quality, best_index = quality, ni
            if _count(ns) <= target:
                return reconstruct(nodes, ni)
            heapq.heappush(heap, (ng+1.8*h+seenpenalty, serial, ni))
            serial += 1
        if len(nodes) > 16000:
            break
    if best_index is not None:
        return reconstruct(nodes, best_index)
    return []


def _ring(obs, p, off):
    x, y = _screen(p, off)
    sx, sy = x-3, y-3
    x0, x1 = max(0, sx), min(64, sx+6)
    y0, y1 = max(1, sy), min(64, sy+6)
    return (x0 < x1 and y0 < y1
            and np.count_nonzero(obs[y0:y1, x0:x1] == 2) >= 3)


def _issue(m, cmd):
    s, off = m["s"], m["off"]
    floor = {p for p, t in m["terrain"].items() if t == 1}
    rails = {p for p, t in m["terrain"].items() if t == 2}
    ns = _transition(s, cmd, floor, rails)
    if ns is None or not _legal_edge(s, ns, off, cmd, m):
        return None
    noff = _next_off(s, ns, off, cmd, m)
    if cmd[0] == "j":
        m["pending"] = cmd
        x, y = _screen(cmd[2], off)
        return "ACTION6", int(y), int(x)
    m["journal"].append((s, off, cmd, ns, 1))
    m["issued"] = (s, off, cmd, ns, noff)
    m["s"], m["off"] = ns, noff
    return _AN[cmd[1]], None, None


def _undo(m, available, block=True):
    if "ACTION7" not in available or not m["journal"]:
        return None
    before, off, cmd, after, count = m["journal"].pop()
    if block and cmd is not None:
        m["bad"].add((before, cmd))
    restored = []
    for i in range(5):
        current = set(m["s"][i])
        current.difference_update(set(after[i])-set(before[i]))
        current.update(set(before[i])-set(after[i]))
        restored.append(tuple(sorted(current)))
    m["s"], m["off"] = tuple(restored), off
    m["plan"] = []
    m.pop("pending", None)
    m.pop("issued", None)
    m["undo_left"] = count-1
    return "ACTION7", None, None


def act(obs, available, memory):
    obs = np.asarray(obs)
    available = set(available)
    fresh, gray, unsafe_pan, issued = _observe(obs, memory)

    left = memory.get("undo_left", 0)
    if left:
        memory["undo_left"] = left-1
        if "ACTION7" in available:
            return "ACTION7", None, None

    if gray or unsafe_pan:
        if gray and issued is not None:
            before, before_off, cmd, predicted, predicted_off = issued
            if cmd[0] == "j":
                memory["traps"][cmd[3]] = max(
                    memory["traps"].get(cmd[3], -1), _count(predicted)
                )
        answer = _undo(memory, available)
        if answer is not None:
            return answer
        if "RESET" in available:
            memory.clear()
            return "RESET", None, None

    pending = memory.pop("pending", None)
    if pending is not None:
        s, off = memory["s"], memory["off"]
        floor = {p for p, t in memory["terrain"].items() if t == 1}
        rails = {p for p, t in memory["terrain"].items() if t == 2}
        ns = _transition(s, pending, floor, rails)
        if (ns is not None and _legal_edge(s, ns, off, pending, memory)
                and _ring(obs, pending[3], off)
                and "ACTION6" in available):
            noff = _next_off(s, ns, off, pending, memory)
            memory["journal"].append((s, off, pending, ns, 2))
            memory["issued"] = (s, off, pending, ns, noff)
            memory["s"], memory["off"] = ns, noff
            x, y = _screen(pending[3], off)
            return "ACTION6", int(y), int(x)
        memory["bad"].add((s, pending))
        memory["plan"] = []
        memory["journal"].append((s, off, None, s, 1))

    if memory["plan"]:
        cmd = memory["plan"].pop(0)
        answer = _issue(memory, cmd)
        if answer is not None and answer[0] in available:
            return answer
        memory["plan"] = []

    s = memory["s"]
    use_macro = (
        bool(s[1]) and not s[2] and len(s[3]) >= 2
        and _count(s) > int(bool(s[0])) + int(bool(s[1]))
        and all(_visible(p, memory["off"]) for p in s[0]+s[1])
    )
    plan = _rail_plan(memory) if use_macro else _plan(memory)
    if plan:
        memory["plan"] = plan[1:]
        answer = _issue(memory, plan[0])
        if answer is not None and answer[0] in available:
            return answer
        memory["plan"] = []

    s, off = memory["s"], memory["off"]
    floor = {p for p, t in memory["terrain"].items() if t == 1}
    rails = {p for p, t in memory["terrain"].items() if t == 2}
    candidates = []
    for ns, cmd, cost in _neighbors(s, floor, rails):
        if not _legal_edge(s, ns, off, cmd, memory):
            continue
        noff = _next_off(s, ns, off, cmd, memory)
        if memory["visits"].get((ns, noff), 0):
            continue
        score = 20*(_count(s)-_count(ns))-cost
        if cmd[0] == "j":
            score += .04*(_xy(cmd[3])[0]-_xy(cmd[2])[0])
        candidates.append((score, cmd))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, cmd in candidates:
        answer = _issue(memory, cmd)
        if answer is not None and answer[0] in available:
            return answer

    answer = _undo(memory, available)
    if answer is not None:
        return answer
    if "RESET" in available:
        memory.clear()
        return "RESET", None, None
    if "ACTION7" in available:
        return "ACTION7", None, None
    for name in _AN:
        if name in available:
            return name, None, None
    return "ACTION6", 32, 32

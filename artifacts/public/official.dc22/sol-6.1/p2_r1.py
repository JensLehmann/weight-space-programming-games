import numpy as np
import heapq
import time
from collections import deque


def _components(mask, diagonal=True):
    a = np.asarray(mask, dtype=bool)
    h, w = a.shape
    seen = np.zeros_like(a)
    out = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for yy, xx in zip(*np.nonzero(a)):
        yy, xx = int(yy), int(xx)
        if seen[yy, xx]:
            continue
        seen[yy, xx] = True
        stack = [(xx, yy)]
        pts = []
        while stack:
            x, y = stack.pop()
            pts.append((x, y))
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and
                        a[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append((min(xs), min(ys), max(xs)+1, max(ys)+1, pts))
    return out


def _bits(a):
    return int.from_bytes(
        np.packbits(np.asarray(a, dtype=np.uint8).ravel(),
                    bitorder="little").tobytes(), "little")


def _blocks(a):
    a = np.asarray(a, dtype=bool)
    b = a.copy()
    b[:-1, :] |= a[1:, :]
    b[:, :-1] |= a[:, 1:]
    b[:-1, :-1] |= a[1:, 1:]
    return _bits(b)


def _rect(x, y, w, h):
    a = np.zeros((64, 64), bool)
    x0, x1 = max(0, x), min(64, x+w)
    y0, y1 = max(0, y), min(64, y+h)
    if x1 > x0 and y1 > y0:
        a[y0:y1, x0:x1] = True
    return a


def _shift(a, dx, dy):
    b = np.zeros((64, 64), bool)
    x0, x1 = max(0, dx), min(64, 64+dx)
    y0, y1 = max(0, dy), min(64, 64+dy)
    if x1 > x0 and y1 > y0:
        b[y0:y1, x0:x1] = a[y0-dy:y1-dy, x0-dx:x1-dx]
    return b


def _crusher_pixels(held):
    rows = (
        ["........", "...#####", ".####..#", "####....",
         "####....", ".####..#", "...#####", "........"]
        if held else
        ["...#####", "...#...#", ".####...", "####....",
         "####....", ".####...", "...#...#", "...#####"]
    )
    return np.array([[c == "#" for c in row] for row in rows], bool)


def _restore_player(image, raw, world, px, py, memory):
    player = raw == 14
    previous = memory.get("ground_image")
    command = memory.get("previous_command")

    # Fixed pads are remembered independently of moving platforms.
    pads = memory.get("fixed_pads")
    if pads is not None:
        image[player & pads] = 2

    # For ordinary movement the terrain has not changed. Copying the
    # previous unobscured pixels is much more reliable than rim voting.
    if previous is not None:
        missing = player & (image == 14)
        suitable = missing & (previous != 14) & (previous != 4)
        if command in ("e", "f"):
            suitable &= (previous != 1) & (previous != 12) & (previous != 5)
        if command in ("up", "down", "left", "right", "grab"):
            suitable &= previous != 12
        image[suitable] = previous[suitable]

    # Recover rectangular fixed pads, including the player's hidden area.
    missing = image == 14
    for x0, y0, x1, y1, pts in _components(
            ((image == 2) | missing) & world, False):
        area = (x1-x0)*(y1-y0)
        if area and len(pts) >= .90*area and np.any(image[y0:y1, x0:x1] == 2):
            patch = image[y0:y1, x0:x1]
            patch[patch == 14] = 2

    # A player on a solid bridge can obscure a quarter of its pixels.
    missing = image == 14
    for color in (9, 15):
        for x0, y0, x1, y1, pts in _components(
                ((image == color) | missing) & world):
            if not np.any(image[y0:y1, x0:x1] == 14):
                continue
            area = (x1-x0)*(y1-y0)
            patch = image[y0:y1, x0:x1]
            colored = np.count_nonzero(patch == color)
            if colored and len(pts) >= .80*area:
                patch[patch == 14] = color

    # For a two-row translating strip, use colors from that row.
    if np.any(image == 14):
        patch = image[max(0, py-1):min(64, py+3),
                      max(0, px-3):min(64, px+5)]
        if command in ("e", "f"):
            colors = patch[(patch == 1) | (patch == 12)]
            if colors.size:
                vals, counts = np.unique(colors, return_counts=True)
                image[image == 14] = int(vals[np.argmax(counts)])

    # Last-resort rim inference, used principally on fresh perturbed starts.
    if np.any(image == 14):
        rim = image[max(0, py-1):min(64, py+3),
                    max(0, px-1):min(64, px+3)]
        vals, nums = np.unique(rim, return_counts=True)
        counts = dict(zip(map(int, vals), map(int, nums)))
        choices = [c for c in (13, 8, 9, 12, 1, 15, 6, 7, 2)
                   if counts.get(c, 0)]
        if choices:
            color = max(choices, key=lambda c: counts[c])
            image[image == 14] = color
        else:
            image[image == 14] = 2

    if pads is None:
        pads = np.zeros((64, 64), bool)
    pads |= (image == 2) & world
    memory["fixed_pads"] = pads


def _scene(obs, memory):
    raw = np.asarray(obs, dtype=np.int16)
    image = raw.copy()
    pcs = _components(raw == 14)
    if not pcs:
        return None
    pc = max(pcs, key=lambda z: len(z[4]))
    px, py = pc[0], pc[1]
    position = py*64+px

    active = np.any(image == 4, axis=1)
    height = max(1, int(active.sum()))
    counts = np.sum((image == 5) & active[:, None], axis=0)
    candidates = [x for x in range(26, 61) if counts[x] > .66*height]
    sidebar = candidates[0] if candidates else 40
    width = max(24, sidebar-2)
    world = np.zeros((64, 64), bool)
    world[:63, :width] = True
    empty = np.zeros((64, 64), bool)

    buttons, arrows = {}, {}
    tens, nines = [], []
    for color, name in ((8, "a"), (9, "b"), (6, "c"),
                        (15, "d"), (2, "e"), (11, "f")):
        for x0, y0, x1, y1, pts in _components((image == color) & ~world):
            w, h = x1-x0, y1-y0
            if color in (8, 9, 6, 15) and w >= 11 and 4 <= h <= 7:
                buttons[name] = ((y0+y1-1)//2, (x0+x1-1)//2)
            elif color == 2 and 5 <= w <= 8 and 5 <= h <= 8:
                buttons["e"] = ((y0+y1-1)//2, (x0+x1-1)//2)
            elif color == 11 and 5 <= w <= 9 and 5 <= h <= 9:
                x, y = pts[len(pts)//2]
                buttons["f"] = (y, x)
            elif color == 8 and 7 <= w <= 10 and 2 <= h <= 3:
                arrows["grab"] = ((y0+y1-1)//2, (x0+x1-1)//2)
            elif color == 9 and w == 4 and h == 4:
                nines.append((x0, y0))
    for x0, y0, x1, y1, pts in _components((image == 10) & ~world):
        if x1-x0 == 4 and y1-y0 == 4:
            tens.append((x0, y0))
    if len(tens) >= 2:
        tens.sort()
        arrows["left"] = (tens[0][1]+1, tens[0][0]+1)
        arrows["right"] = (tens[-1][1]+1, tens[-1][0]+1)
    if len(nines) >= 2:
        nines.sort()
        arrows["up"] = (nines[0][1]+1, nines[0][0]+1)
        arrows["down"] = (nines[-1][1]+1, nines[-1][0]+1)
    known_buttons = memory.setdefault("buttons", {})
    known_buttons.update(buttons)

    goal = None
    for x0, y0, x1, y1, pts in _components((image == 11) & world):
        if x1-x0 <= 3 and y1-y0 <= 3:
            goal = y0*64+x0
            break
    if goal is not None:
        memory["goal"] = goal
    else:
        goal = memory.get("goal")

    _restore_player(image, raw, world, px, py, memory)

    portals = memory.setdefault("portals", {})
    if memory.pop("flip_portals", False):
        for p in list(portals):
            portals[p] = 13-portals[p]
    for x0, y0, x1, y1, pts in _components(
            ((image == 6) | (image == 7)) & world):
        if x1-x0 == 2 and y1-y0 == 2:
            portals[y0*64+x0] = int(image[y0, x0])
    if position in portals:
        c = portals[position]
        image[py:py+2, px:px+2] = [[c, 13-c], [13-c, c]]

    floor = ((image == 2) | (image == 11) |
             (image == 6) | (image == 7)) & world
    floor |= memory["fixed_pads"] & world
    for p in portals:
        x, y = p % 64, p//64
        floor[y:y+2, x:x+2] = True

    groups = {}
    keys = []

    def add(name, supports, walls):
        if name not in groups:
            groups[name] = [[], []]
        groups[name][0].append(supports)
        groups[name][1].append(walls)

    # Checkerboard and solid bridges.
    for color, name in ((9, "b"), (15, "d")):
        for x0, y0, x1, y1, pts in _components((image == color) & world):
            w, h = x1-x0, y1-y0
            if name == "d" and w <= 4 and h == 2 and len(pts) <= 7:
                keys.append((x0, y0, x1, y1))
                continue
            if w < 2 or h < 2:
                continue
            rect = _rect(x0, y0, w, h)
            checker = empty.copy()
            parity = (pts[0][0]+pts[0][1]) & 1
            yy, xx = np.mgrid[y0:y1, x0:x1]
            checker[y0:y1, x0:x1] = ((xx+yy) & 1) == parity
            if len(pts) > .60*w*h:
                add(name, [rect, empty], [empty, checker])
            else:
                add(name, [empty, rect], [checker, empty])

    # Blue pivot platforms and machinery.
    crusher = None
    for x0, y0, x1, y1, pts in _components(
            ((image == 8) | (image == 13)) & world):
        pivots = [(x, y) for x, y in pts if image[y, x] == 13]
        if pivots:
            xs, ys = [p[0] for p in pivots], [p[1] for p in pivots]
            cx2, cy2 = min(xs)+max(xs), min(ys)+max(ys)
            a, b = empty.copy(), empty.copy()
            for x, y in pts:
                a[y, x] = True
                nx = (cx2+2*y-cy2)//2
                ny = (cy2+2*x-cx2)//2
                if 0 <= nx < 64 and 0 <= ny < 64:
                    b[ny, nx] = True
            add("a", [a, b], [empty, empty])
        elif arrows and x1-x0 >= 6 and y1-y0 >= 5:
            held = x1-x0 == 8 and y1-y0 == 6
            known = x1-x0 == 8 and y1-y0 in (6, 8)
            oy = y0-1 if held and known else y0
            mask = empty.copy()
            if known and 0 <= oy <= 56 and x0 <= 56:
                mask[oy:oy+8, x0:x0+8] = _crusher_pixels(held)
            else:
                for x, y in pts:
                    mask[y, x] = True
            crusher = (x0, oy, mask, held, known,
                       x1-x0, 8 if known else y1-y0)
        else:
            for x, y in pts:
                floor[y, x] = True

    objects, thin = [], []
    for x0, y0, x1, y1, pts in _components(
            ((image == 1) | (image == 12)) & world):
        w, h = x1-x0, y1-y0
        if h <= 2:
            thin.append((x0, y0, x1, y1, pts))
        else:
            a = empty.copy()
            if w == 6 and h == 12:
                local = np.ones((12, 6), bool)
                local[:, :2] = False
                local[4, 0] = local[7, 0] = True
                local[5:7, :2] = True
                a[y0:y1, x0:x1] = local
            else:
                for x, y in pts:
                    a[y, x] = True
            objects.append((a, (x0+x1)//2, (y0+y1)//2))

    # Extendable and translating strips.
    strip_memory = memory.setdefault("strips", {})
    blanks = []
    for x0, y0, x1, y1, pts in _components((image == 5) & world, False):
        if x1-x0 == 16 and y1-y0 == 2:
            blanks.append((x0, y0))

    clusters = []
    thin.sort(key=lambda z: (z[1], z[0]))
    for item in thin:
        if (clusters and clusters[-1][1] == item[1] and
                clusters[-1][3] == item[3] and item[0]-clusters[-1][2] <= 12):
            old = clusters[-1]
            clusters[-1] = (old[0], old[1], item[2], old[3],
                            old[4]+item[4])
        else:
            clusters.append(item)

    descriptors = []
    for x, y in blanks:
        if "e" in known_buttons:
            descriptors.append(("e", x, y, 0))
            strip_memory[("e", y)] = (x, 0)

    previous_command = memory.get("previous_command")
    for x0, y0, x1, y1, pts in clusters:
        w = x1-x0
        orange = any(image[y, x] == 12 for x, y in pts)
        known_e = strip_memory.get(("e", y0))
        known_f = strip_memory.get(("f", y0))
        if (w >= 12 or known_e is not None or
                ("e" in known_buttons and "f" not in known_buttons)):
            name = "e"
            previous = known_e
            bx = previous[0] if previous else x0
            phase = 4 if orange else min(3, max(1, len(pts)//4))
            if previous:
                guess = previous[1]
                if previous_command == "e":
                    guess = (guess+1) % 8
                edge = min(guess, 8-guess)
                if (guess == 4 and orange) or (not orange and edge == phase):
                    phase = guess
            descriptors.append((name, bx, y0, phase))
            strip_memory[(name, y0)] = (bx, phase)
        elif "f" in known_buttons:
            name = "f"
            previous = known_f
            if orange:
                bx, phase = (previous[0] if previous else x0-10), 5
            elif previous:
                bx = previous[0]
                guess = previous[1]
                if previous_command == "f":
                    guess = (guess+1) % 6
                # Use the predicted phase if all visible pixels fit it.
                if all(bx+2*guess <= x < bx+2*guess+6 for x, y in pts):
                    phase = guess
                else:
                    phase = max(0, min(5, (x0-bx)//2))
            else:
                near = sorted(blanks, key=lambda p:
                              abs(p[0]-x0)+abs(p[1]-y0))
                bx = near[0][0] if near and abs(near[0][0]-x0) <= 10 else x0
                phase = max(0, min(5, (x0-bx)//2))
            descriptors.append((name, bx, y0, phase))
            strip_memory[(name, y0)] = (bx, phase)
        else:
            for x, y in pts:
                floor[y, x] = True

    for name, bx, y, phase in descriptors:
        variants = []
        number = 8 if name == "e" else 6
        for j in range(number):
            k = (phase+j) % number
            a = empty.copy()
            if name == "e":
                if k == 4:
                    a |= _rect(bx, y, 16, 2)
                else:
                    edge = 2*min(k, 8-k)
                    if edge:
                        a |= _rect(bx, y, edge, 2)
                        a |= _rect(bx+16-edge, y, edge, 2)
            else:
                a |= _rect(bx+2*k, y, 6, 2)
            variants.append(a)
        add(name, variants, [empty]*number)

    compiled = {}
    for name, (spieces, wpieces) in groups.items():
        number = max(len(v) for v in spieces)
        variants = []
        for k in range(number):
            support, wall = empty.copy(), empty.copy()
            for piece in spieces:
                support |= piece[k % len(piece)]
            for piece in wpieces:
                wall |= piece[k % len(piece)]
            variants.append((_bits(support), _blocks(wall)))
        compiled[name] = variants

    machinery = None
    if crusher:
        cx, cy, mask, held, known, cw, ch = crusher
        origin = memory.get("crusher_origin")
        if origin is None and known:
            pts = [p for c in _components((image == 0) & world) for p in c[4]]
            if pts:
                ox, oy = min(p[0] for p in pts)-1, max(p[1] for p in pts)-5
                if ((cx-ox) % 4 == 0 and (oy-cy) % 4 == 0 and
                        0 <= (cx-ox)//4 <= 3 and 0 <= (oy-cy)//4 <= 3):
                    origin = (ox, oy)
        if origin is None:
            origin = (cx, cy)
        memory["crusher_origin"] = origin
        ox, oy = origin
        if known:
            positions = [(ox, oy-4*j) for j in range(4)]
            positions += [(ox+4*i, oy-4*j)
                          for j in (0, 3) for i in range(1, 4)]
        else:
            positions = []
            for y in range(cy % 4, 60, 4):
                for x in range(cx % 4, max(1, width-4), 4):
                    ax, ay = x+cw//2, y+ch//2
                    if (0 <= ax < width and 0 <= ay < 63 and
                            (image[ay, ax] == 0 or (x == cx and y == cy))):
                        positions.append((x, y))
        if (cx, cy) not in positions:
            positions.append((cx, cy))
        root_index = positions.index((cx, cy))
        offset = (9, 4) if known else (cw//2, ch//2)
        carried = None
        if objects:
            carried = min(objects, key=lambda o:
                          abs(o[1]-cx-offset[0])+abs(o[2]-cy-offset[1]))
        for obj in objects:
            if obj is not carried:
                floor |= obj[0]

        crusher_walls, object_supports = [], []
        for x, y in positions:
            crusher_walls.append(_blocks(_shift(mask, x-cx, y-cy)))
            if carried:
                object_supports.append(_bits(_shift(
                    carried[0], x+offset[0]-carried[1],
                    y+offset[1]-carried[2])))
            else:
                object_supports.append(0)
        initial_object = _bits(carried[0]) if carried else 0
        grabs = set()
        if carried:
            for i, (x, y) in enumerate(positions):
                if (x+offset[0], y+offset[1]) == carried[1:]:
                    grabs.add(i)
        links = {}
        index = {p: i for i, p in enumerate(positions)}
        for i, (x, y) in enumerate(positions):
            for command, dx, dy in (("up", 0, -4), ("down", 0, 4),
                                    ("left", -4, 0), ("right", 4, 0)):
                if command in arrows and (x+dx, y+dy) in index:
                    links.setdefault(i, []).append((command, index[x+dx, y+dy]))
        machinery = (root_index, int(held), crusher_walls,
                     object_supports, initial_object, grabs, links)
    else:
        for obj in objects:
            floor |= obj[0]

    memory["ground_image"] = image.copy()
    names = sorted(compiled)
    root = tuple([0]*len(names) +
                 ([machinery[0], machinery[1]] if machinery else [0, 0]) +
                 [int("d" in buttons and not keys)])
    return {
        "position": position, "goal": goal, "width": width,
        "buttons": buttons, "arrows": arrows, "groups": compiled,
        "names": names, "support": _bits(floor), "walls": 0,
        "portals": dict(portals), "keys": keys,
        "machine": machinery, "root": root,
        "denied": memory.setdefault("denied", {})
    }


def _neighbors(p, width):
    x, y = p % 64, p//64
    if y >= 2:
        yield p-128, "ACTION1"
    if y < 61:
        yield p+128, "ACTION2"
    if x >= 2:
        yield p-2, "ACTION3"
    if x+2 < width:
        yield p+2, "ACTION4"


def _raw_geometry(scene, config):
    s, w = scene["support"], scene["walls"]
    names = scene["names"]
    for i, name in enumerate(names):
        a, b = scene["groups"][name][config[i]]
        s |= a
        w |= b
    machine = scene["machine"]
    if machine:
        n = len(names)
        idx, held = config[n], config[n+1]
        w |= machine[2][idx]
        s |= machine[3][idx] if held else machine[4]
    return s & ~w


def _geometry(scene, config):
    v = _raw_geometry(scene, config)
    return v & ~scene["denied"].get(v, 0)


def _changes(scene, config):
    names, groups = scene["names"], scene["groups"]
    n = len(names)
    for i, name in enumerate(names):
        enabled = name in scene["buttons"]
        if name == "d" and config[-1]:
            enabled = True
        if enabled and len(groups[name]) > 1:
            nxt = list(config)
            nxt[i] = (nxt[i]+1) % len(groups[name])
            yield name, tuple(nxt)
    machine = scene["machine"]
    if machine:
        idx, held = config[n], config[n+1]
        for command, dest in machine[6].get(idx, []):
            nxt = list(config)
            nxt[n] = dest
            yield command, tuple(nxt)
        if not held and idx in machine[5] and "grab" in scene["arrows"]:
            nxt = list(config)
            nxt[n+1] = 1
            yield "grab", tuple(nxt)


def _plan(scene, memory):
    start, goal = scene["position"], scene["goal"]
    root, names = scene["root"], scene["names"]
    groups, machine = scene["groups"], scene["machine"]
    width = scene["width"]
    deadline = time.monotonic()+6.0
    if goal is None:
        return []

    if not ((_geometry(scene, root) >> start) & 1):
        scene["support"] |= 1 << start

    relaxed = scene["support"]
    for variants in groups.values():
        for s, w in variants:
            relaxed |= s
    if machine:
        relaxed |= machine[4]
        for s in machine[3]:
            relaxed |= s
    relaxed &= ~scene["walls"]

    portal_edges = {}
    portals = scene["portals"]
    for p, color in portals.items():
        targets = [q for q, c in portals.items() if q != p and c != color]
        if not targets:
            targets = [q for q in portals if q != p]
        if targets:
            portal_edges[p] = targets[0]
    reverse_portals = {}
    for a, b in portal_edges.items():
        reverse_portals.setdefault(b, []).append(a)

    heuristic = {goal: 0}
    queue = deque([goal])
    while queue:
        p = queue.popleft()
        d = heuristic[p]+1
        for q, command in _neighbors(p, width):
            if q not in heuristic and ((relaxed >> q) & 1):
                heuristic[q] = d
                queue.append(q)
        for q in reverse_portals.get(p, []):
            if q not in heuristic:
                heuristic[q] = d
                queue.append(q)

    def h(p):
        if p in heuristic:
            return heuristic[p]
        return (abs(p % 64-goal % 64)+abs(p//64-goal//64))/2+20

    parity_mask = 0
    parities = {(start % 2, (start//64) % 2)}
    for p in portals:
        parities.add((p % 2, (p//64) % 2))
    for xp, yp in parities:
        for y in range(yp, 63, 2):
            for x in range(xp, width, 2):
                parity_mask |= 1 << (64*y+x)

    cache = {}

    def geometry(config):
        if config in cache:
            return cache[config]
        valid = _geometry(scene, config)
        rest = valid & parity_mask
        positions = set()
        while rest:
            low = rest & -rest
            positions.add(low.bit_length()-1)
            rest -= low
        labels, components = {}, {}
        while positions:
            seed = min(positions)
            positions.remove(seed)
            stack = [seed]
            pts = []
            while stack:
                p = stack.pop()
                labels[p] = seed
                pts.append(p)
                for q, command in _neighbors(p, width):
                    if q in positions:
                        positions.remove(q)
                        stack.append(q)
            components[seed] = pts
        result = valid, labels, components
        cache[config] = result
        return result

    valid, labels, comps = geometry(root)
    if start not in labels:
        return []
    cid = labels[start]
    first = {"config": root, "pos": start, "cid": cid, "cost": 0,
             "parent": None, "cross": None, "command": None}
    costs = {(root, cid): 0}
    serial = 0
    heap = [(min(h(p) for p in comps[cid]), 0, serial, first)]

    visits = memory.setdefault("macro_visits", {})
    signature = (valid, start)
    visits[signature] = visits.get(signature, 0)+1
    best_record, best_endpoint = None, None
    best_score = float("inf")
    expanded = 0

    def flood(record):
        config, p = record["config"], record["pos"]
        allowed = set(geometry(config)[2][record["cid"]])
        dist = {p: 0}
        tree = {}
        todo = deque([p])
        while todo:
            p = todo.popleft()
            for q, command in _neighbors(p, width):
                if q in allowed and q not in dist:
                    dist[q] = dist[p]+1
                    tree[q] = (p, command)
                    todo.append(q)
        record["tree"] = tree
        return dist

    def push(parent, cross, command, new_pos, config, extra):
        nonlocal serial
        v, labs, components = geometry(config)
        if new_pos not in labs:
            return
        new_cid = labs[new_pos]
        key = (config, new_cid)
        cost = parent["cost"]+extra
        if cost >= costs.get(key, 10**9):
            return
        costs[key] = cost
        record = {"config": config, "pos": new_pos, "cid": new_cid,
                  "cost": cost, "parent": parent, "cross": cross,
                  "command": command}
        serial += 1
        estimate = min(h(p) for p in components[new_cid])
        heapq.heappush(heap, (cost+1.5*estimate, cost, serial, record))

    found_record, found_endpoint = None, None
    while heap and expanded < 40000:
        if expanded % 32 == 0 and time.monotonic() > deadline:
            break
        _, cost, _, record = heapq.heappop(heap)
        config, cid = record["config"], record["cid"]
        if costs.get((config, cid)) != cost:
            continue
        expanded += 1
        dist = flood(record)
        v = geometry(config)[0]
        if goal in dist:
            found_record, found_endpoint = record, goal
            break

        for p in dist:
            score = (1.5*h(p)+.12*(cost+dist[p]) +
                     6*visits.get((v, p), 0))
            if not root[-1] and config[-1]:
                score -= 7
            if cost+dist[p] > 0 and score < best_score:
                best_score = score
                best_record, best_endpoint = record, p

        for command, nxt in _changes(scene, config):
            nv, labs, components = geometry(nxt)
            choices = {}
            for p, d in dist.items():
                new_cid = labs.get(p)
                if new_cid is not None:
                    old = choices.get(new_cid)
                    if old is None or d < old[0]:
                        choices[new_cid] = (d, p)
            for d, p in choices.values():
                push(record, p, command, p, nxt, d+1)

        if not config[-1]:
            key_positions = []
            for p, d in dist.items():
                x, y = p % 64, p//64
                if any(x+1 >= x0 and y+1 >= y0 and x < x1 and y < y1
                       for x0, y0, x1, y1 in scene["keys"]):
                    key_positions.append((d, p))
            if key_positions:
                d, p = min(key_positions)
                push(record, p, None, p, config[:-1]+(1,), d)

        if "c" in scene["buttons"]:
            for p, target in portal_edges.items():
                if p in dist:
                    push(record, p, "c", target, config, dist[p]+1)

    record = found_record or best_record
    endpoint = found_endpoint if found_record else best_endpoint
    if record is None:
        return []

    def walking(rec, target):
        result = []
        p = target
        tree = rec["tree"]
        while p != rec["pos"]:
            if p not in tree:
                return []
            prev, command = tree[p]
            result.append((command, p, geometry(rec["config"])[0]))
            p = prev
        result.reverse()
        return result

    chain = []
    current = record
    while current["parent"] is not None:
        chain.append(current)
        current = current["parent"]
    chain.reverse()
    plan = []
    for child in chain:
        parent = child["parent"]
        plan.extend(walking(parent, child["cross"]))
        if child["command"] is not None:
            plan.append((child["command"], child["pos"],
                         geometry(child["config"])[0]))
    plan.extend(walking(record, endpoint))
    if not found_record:
        plan = plan[:12]
    return plan


def act(obs, available, memory):
    allowed = set(available)
    scene = _scene(obs, memory)
    if scene is None:
        memory.pop("plan", None)
        memory.pop("expected", None)
        action = ("RESET" if "RESET" in allowed else
                  next((a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
                        if a in allowed), available[0]))
        return action, None, None

    if scene["position"] == scene["goal"]:
        return next((a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
                     if a in allowed), "RESET"), None, None

    current_valid = _geometry(scene, scene["root"])
    expected = memory.pop("expected", None)
    plan = memory.get("plan", [])
    if expected is not None:
        predicted_pos, predicted_valid = expected
        mismatch = predicted_pos != scene["position"]
        if mismatch and memory.get("previous_command", "").startswith("ACTION"):
            old_valid = memory.get("issued_geometry")
            if old_valid is not None and predicted_pos != scene["position"]:
                denied = memory.setdefault("denied", {})
                denied[old_valid] = denied.get(old_valid, 0) | (1 << predicted_pos)
                current_valid = _geometry(scene, scene["root"])
        if predicted_valid is not None:
            if (predicted_valid ^ current_valid).bit_count() > 8:
                mismatch = True
        if mismatch:
            plan = []

    if plan:
        command, next_pos, predicted_valid = plan[0]
        if command.startswith("ACTION"):
            if not ((current_valid >> next_pos) & 1):
                plan = []
        elif (command not in scene["buttons"] and
              command not in scene["arrows"]):
            plan = []

    if not plan:
        plan = _plan(scene, memory)
        current_valid = _geometry(scene, scene["root"])

    if not plan:
        # Only probe modeled switches if the player remains supported.
        probes = memory.setdefault("probes", {})
        candidates = []
        p = scene["position"]
        for command, config in _changes(scene, scene["root"]):
            if command not in scene["buttons"] and command not in scene["arrows"]:
                continue
            v = _geometry(scene, config)
            if not ((v >> p) & 1):
                continue
            if v == current_valid:
                continue
            key = (current_valid, p, command)
            candidates.append((probes.get(key, 0), command, key, v))
        if "c" in scene["buttons"] and p in scene["portals"]:
            key = (current_valid, p, "c")
            candidates.append((probes.get(key, 0), "c", key, None))
        if candidates:
            _, command, key, v = min(candidates, key=lambda z: (z[0], z[1]))
            probes[key] = probes.get(key, 0)+1
            plan = [(command, p, v)]
        else:
            visits = memory.setdefault("fallback_visits", {})
            choices = []
            for q, command in _neighbors(p, scene["width"]):
                if command in allowed and ((current_valid >> q) & 1):
                    choices.append((visits.get(q, 0), command, q))
            if choices:
                _, command, q = min(choices)
                visits[q] = visits.get(q, 0)+1
                plan = [(command, q, current_valid)]
            else:
                return next((a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
                             if a in allowed), "RESET"), None, None

    command, next_pos, predicted_valid = plan.pop(0)
    memory["plan"] = plan
    memory["expected"] = (next_pos, predicted_valid)
    memory["previous_command"] = command
    memory["issued_geometry"] = _raw_geometry(scene, scene["root"])

    if command.startswith("ACTION"):
        if command not in allowed:
            memory["plan"] = []
            memory.pop("expected", None)
            return next((a for a in available if a != "ACTION6"), "RESET"), None, None
        return command, None, None

    location = scene["buttons"].get(command, scene["arrows"].get(command))
    if location is None or "ACTION6" not in allowed:
        memory["plan"] = []
        memory.pop("expected", None)
        return next((a for a in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
                     if a in allowed), "RESET"), None, None
    if command == "c":
        memory["flip_portals"] = True
    row, col = location
    return "ACTION6", int(row), int(col)
